"""
クリップ幅推定(C2) :MAD+EMA+P95 バックストップ

目的
____
- 量子化前の層差分 u_ℓのスケールを外れ値に頑強に数いていし、クリップ幅 c_ℓを時間的に安定させる

手法
____
- 初期: `c = k * MAD(u)` (k≒2.5)
- 更新: `c ← (1-ξ)·c + ξ·c_obs`（EMA, クリップ観測 c_obs は都度の MAD ベース）
- バックストップ: `c ← max(c, β_c * P95(|u|))`（下振れで飽和しないよう保険）
- 前半ラウンドのみ高めの ξ を使う **ウォームアップ**（例: ξ=0.3, ステップ10）
"""

from __future__ import annotations
from typing import Dict

import torch
from ..interfaces import IClipper, LayerId, Tensor

class ClipperMADEMA(IClipper):
    """MAD+EMAによるロバストなクリップ幅推定器

    Parameters
    ___________
    k: float
        初期/観測時のスケール係数
    xi: float
        通常時のEMA係数
    xi_warmup: float
        ウォームアップ期間中のEMA係数
    warmup_steps: int
        ウォームアップとして高いξ を使う観測回数（層ごとにカウント）
    beta_c : float
        P95 バックストップの係数（例: 0.8）
    eps : float
        数値安定用の微小値
    """
    def __init__(self, *, k: float=2.5, xi: float=0.1, xi_warmup: float=0.3, warmup_steps: int=10, beta_c: float=0.8, eps: float=1e-12)->None:
        self.k = float(k)
        self.xi = float(xi)
        self.xi_warmup = float(xi_warmup)
        self.xi_warmup_steps = int(warmup_steps)
        self.beta_c = float(beta_c)
        self.eps = float(eps)

        self._c: Dict[LayerId, float] = {}
        self._obs: Dict[LayerId, int] = {}

    @torch.no_grad()
    def observe(self, layer: LayerId, update_tensor: Tensor) ->None:
        """
        観測テンソルからc_ℓを更新

        - 観測ベース: MAD(u)とP95(|u|)
        - EMA: c←(1-ξ)c + ξ·(k*MAD)
        - バックストップ: c ← max(c, β_c*P95)
        """
        x = update_tensor.detach().to(dtype=torch.float64, device="cpu").view(-1)
        if x.numel() == 0:
            return

        # MAD: median(|x-median(x)|)
        med = torch.median(x)
        mad = torch.median(torch.abs(x - med)) + self.eps
        c_obs = float(self.k * mad)

        # P95(|x|)
        p95 = float(torch.quantile(torch.abs(x), 0.95))

        # 初期 or EMA更新
        if layer not in self._c:
            c = c_obs
            self._obs[layer] = 0
        else:
            steps = self._obs[layer]
            xi_eff = self.xi_warmup if steps < self.xi_warmup_steps else self.xi
            c = (1.0 - xi_eff) * self._c[layer] + xi_eff * c_obs

        # バックストップ
        c = max(c, self.beta_c *p95, self.eps)

        # 保存
        self._c[layer] = c
        self._obs[layer] = self._obs.get(layer, 0) + 1

    def get(self, layer:LayerId) -> float:
        """現在の c_ℓを返す。未観測なら1.0"""
        return float(self._c.get(layer, 1.0))

__all__ = ["ClipperMADEMA"]
