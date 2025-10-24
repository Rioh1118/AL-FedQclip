"""
サーバー側 Clipped 更新(C7) : Basic 実装
======================================

目的
_____
- 集約後の大域更新 (G_t = {g_ell}) のノルムを制御し、
　勾配爆発・外れ値クライアントの影響を抑える。

  数式
  ____
  - 係数:
    H_t = min{η_s, (γ_s * η_s)/(||G_t||_2 + ε)}
  - 適用:
    g'_ℓ = H_t * g_ℓ (任意のℓ)
"""
from __future__ import annotations
from typing import Mapping, Dict

import torch

from ..interfaces import IServerClipper, LayerId, Tensor

class BasicServerClipper(IServerClipper):
    """サーバ側　Clipped更新の汎用版(ステートレス)

    Parameters
    __________
    mode: {"global_l2", "per_layer_l2"}
        クリップの基準ノルムを全層連結にするか、層別にするか
    policy: {"cap_eta", "trust_ratio"}
    eta_s: float
        サーバ側の基準スケール(学習率に相当)
    gamma_s: float
        クリップ強度(大きいほど緩い)
    eps: float
        数値安定用の微小値
    """
    def __init__(self, *, eta_s: float=1.0, gamma_s: float=1.0, eps: float=1e-12)->None:
        self.eta_s = float(eta_s)
        self.gamma_s = float(gamma_s)
        self.eps = float(eps)

    @torch.no_grad()
    def apply(self, *,global_grad: Mapping[LayerId, Tensor])->Mapping[LayerId, Tensor]:
        """大域更新を一括スケールして返す。

        Parameters
        __________
        global_grad: Mapping[LayerId, Tensor
            層ごとの集約更新ベクトル g_ℓ。

        Returns
        _______
        Mapping[LayerId, Tensor]
            Clipped後のg'_ℓ。
        """
        if not global_grad:
            return {}

        # 全層の2乗ノルムを合算して ||G_t||_2を計算
        # dtypeはdoubleで計算して数値安定性を確保
        sq_sum=0.0
        for g in global_grad.values():
            # 各テンソルの2乗和(デバイスは問わず detach-> CPU double に転送して安全にitem取得)
            sq_sum += float(g.detach().to(dtype=torch.float64).pow(2).sum().item())
        g_norm = sq_sum ** 0.5

        # 係数 H_tの計算
        # - g_norm=0の時(γ_s*η_s)/(ε)は大、よってminによりH_t=η_sになる
        H = min(self.eta_s, (self.gamma_s * self.eta_s) / (g_norm + self.eps)) if self.eta_s != 0.0 else 0.0

        # スケーリングして返す(デバイス/dtypeは各層テンソルに合わせてそのまま)
        out: Dict[LayerId, Tensor] = {lid: (g*H) for lid, g in global_grad.items()}
        return out

