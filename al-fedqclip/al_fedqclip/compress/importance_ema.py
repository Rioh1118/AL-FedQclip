"""
層重要度推定(C1): 軽量EMA＋スカラー分散+τ=1+固定ε
===========================================

狙い
____
- FedQClip/FedLP-QのKKT構造(順位依存)を維持しつつ、端末/DP/SecAgg　制約下で安定に動く推定器
- 規定値は"軽量"(ベクトル非保持・固定ε・τ=1)。必要時のみ拡張をONにできる設計

本クラスの主眼
_______________
1) **EMA (ξ)とウォームアップξ_warm** による初期遅れ吸収（補題2）
2) **スカラー分散**: Var ≈ E||g||² − ||E[g]||² を非負に切り上げ（補題4）
3) **τ=1固定**: 時間平滑はスケジューラ側で（命題5）
4) **固定ε**: DP/SecAgg 下での一貫性（§4）
5) **最小拡張（任意）**: debias（短期のみ），log領域ロバスト，Hessianの低頻度注入

- mode="scalar"（既定）: ベクトルの平均は保持せず、||E[g]||² の近似として
"EMAベクトル" を保持するかどうかを切替可能。
- keep_mean_vector=False（既定）: メモリを節約、mu² を近似（EMAの遅れは命題1で吸収）。
- keep_mean_vector=True : ベクトル m_ℓ を保持し、mu²=||m̂||² を直接評価（より精密）。
"""

from __future__ import annotations
from typing import Dict,Mapping,Optional

import math
import torch

from ..interfaces import IImportanceEstimator, LayerId, Tensor,LayerStats

class ImportanceEMA(IImportanceEstimator):
    def __init__(self,
                 *,
                 # 幾何合成の指数
                 alpha:float = 0.4,
                 beta: float =0.0,
                 gamma: float=0.2,
                 # EMAとウォームアップ
                 xi: float=0.1,
                 xi_warm: float = 0.3,
                 warmup_steps: int=10,
                 # Debias
                 use_debias: bool = False,
                 debias_horizon: int = 50,
                 # SNR安定用ε
                 eps: float = 1e-12,
                 eps_warm: Optional[float] = None,
                 eps_switch_round: Optional[int]=None,
                 # メモリと精度のトレードオフ
                 mode: str = "scalar",
                 keep_mean_vector: bool = False,
                 # ロバスト合成
                 log_robust: bool = False,
                ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.xi = float(xi)
        self.xi_warm = float(xi_warm)
        self.warmup_steps = int(warmup_steps)
        self.use_debias = bool(use_debias)
        self.debias_horizon = int(debias_horizon)
        self.eps = float(eps)
        self.eps_warm = float(eps_warm) if eps_warm is not None else self.eps
        self.eps_switch_round = eps_switch_round
        self.mode = mode
        self.keep_mean_vector = bool(keep_mean_vector)
        self.log_robust = bool(log_robust)

        # 状態
        self._t: int = 0 # グローバル更新回数
        self._obs: Dict[LayerId, int] = {}
        self._fisher: Dict[LayerId, float] = {}
        self._mu2: Dict[LayerId, float] = {}
        self._var: Dict[LayerId, float] = {}
        self._mean_vec:Dict[LayerId, Tensor] = {}
        self._htr: Dict[LayerId,float] = {}

    # 更新
    @torch.no_grad()
    def update_from_grads(self, grads: Mapping[LayerId, Tensor]) -> None:
        self._t += 1
        for lid, g in grads.items():
            x = g.detach().to(dtype=torch.float64, device="cpu").view(-1)
            if x.numel() == 0:
                continue
            obs = self._obs.get(lid, 0)
            xi_eff = self.xi_warm if obs < self.warmup_steps else self.xi

            # E||g||² の EMA（fisher 近似）
            g2 = float(torch.sum(x*x).item())
            self._fisher[lid] = self._ema(self._fisher.get(lid, g2), g2, xi_eff, lid)

            #　平均ベクトル
            if self.keep_mean_vector:
                m_prev = self._mean_vec.get(lid)
                if m_prev is None:
                    self._mean_vec[lid] = x.clone()
                else:
                    self._mean_vec[lid] = (1.0 - xi_eff) * m_prev + xi_eff * x
                mu2 = float(torch.sum(self._mean_vec[lid] * self._mean_vec[lid]).item())
            else:
                mu2 = 0.0
            self._mu2[lid] = self._ema(self._mu2.get(lid, mu2), mu2, xi_eff, lid)

            # Var ≈ E||g||² − ||E[g]||²（非負）
            var_est = max(self._fisher[lid] - self._mu2[lid], 0.0)
            self._var[lid] = self._ema(self._var.get(lid, var_est), var_est, xi_eff, lid)

            self._obs[lid] = obs + 1

    @torch.no_grad()
    def update_from_states(self, stats: Mapping[LayerId, LayerStats]) -> None:
        # 外部から安全集約されたスカラーを注入
        self._t += 1
        for lid, st in stats.items():
            obs = self._obs.get(lid, 0)
            xi_eff = self.xi_warm if obs < self.warmup_steps else self.xi
            if hasattr(st, "fisher_ema") and st.fisher_ema is not None:
                v = float(st.fisher_ema)
                self._fisher[lid] = self._ema(self._fisher.get(lid, v), v, xi_eff, lid)
            if hasattr(st, "snr_num") and st.snr_num is not None:
                mu2 = float(st.snr_num)
                self._mu2[lid] = self._ema(self._mu2.get(lid, mu2), mu2, xi_eff, lid)
            if hasattr(st, "snr_den") and st.snr_den is not None:
                varv = max(float(st.snr_den), 0.0)
                self._var[lid] = self._ema(self._var.get(lid, varv), varv, xi_eff, lid)
            if hasattr(st, "hess_trace_norm") and st.hess_trace_norm is not None:
                self._htr[lid] = float(st.hess_trace_norm)
            self._obs[lid] = obs + 1

    #--出力-------------------------------------
    def get_normalized_weights(self, d: Optional[Mapping[LayerId,int]]=None, *, size_weighting: bool = False,) -> Dict[LayerId, float]:
        lids = set(self._fisher.keys()) | set(self._mu2.keys()) | set(self._var.keys()) | set(self._htr.keys())
        if not lids:
            return {}

        # εの切り替え（序盤のみ大きく→一度だけ減らす）
        eps_eff = self._current_eps()

        # 合成(τ=1固定) logg_robustは数値安定の単調変換
        tilde: Dict[LayerId, float]={}
        for lid in lids:
            fisher = max(self._fisher.get(lid, 0.0), 0.0) + eps_eff
            mu2 = max(self._mu2.get(lid, 0.0), 0.0)
            varv = max(self._var.get(lid, max(fisher - mu2, 0.0)), 0.0) + eps_eff
            snr = mu2 / varv
            htr = max(self._htr.get(lid,1.0), eps_eff)

            if self.log_robust:
                val = math.exp(self.alpha * math.log(fisher) + self.gamma * math.log(max(snr, eps_eff)) + self.beta * math.log(htr))
            else:
                val = (fisher ** self.alpha) * (snr ** self.gamma) * (htr ** self.beta)

            tilde[lid] = float(val)

        # 正規化
        total = sum(tilde.values())
        if total <= 0.0:
            uni = 1.0 / len(tilde)
            w = {lid: uni for lid in tilde}
        else:
            w = {lid: v/total for lid, v in tilde.items()}

        if size_weighting and d is not None:
            # W_ℓ = d_ℓ・w_ℓ
            Z = sum(d.get(l, 1)*w.get(l, 0.0) for l in w)
            if Z <= 0.0:
                return w
            return {l: (d.get(l,1) * w[l]) / Z for l in w}
        return w

    def _ema(self, prev: float, new: float, xi_eff: float, lid: LayerId) -> float:
        m = (1.0 - xi_eff) * float(prev) + xi_eff * float(new)
        if self.use_debias:
            t = self._obs.get(lid, 0) + 1
            # 短期のみのdebia
            if t <= self.debias_horizon:
                t = self._obs.get(lid, 0) + 1
                if t <= self.debias_horizon:
                    bias = 1.0 - (1.0 - xi_eff) ** t
                    if bias < 1e-6:
                        m = m/ bias
        return m

    def _current_eps(self) -> float:
        if self.eps_switch_round is None:
            return self.eps
        return self.eps_warm if (self._t <= int(self.eps_switch_round)) else self.eps

__all__ = ["ImportanceEMA"]



