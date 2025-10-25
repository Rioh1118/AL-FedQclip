# -*- coding: utf-8 -*-
# al_fedqclip/compress/bitalloc_allocator.py
from __future__ import annotations
from typing import Dict, Mapping, Optional, Tuple, List
import math

from al_fedqclip.interfaces import LayerId, IBitAllocator


def compute_A_from_clip(
    *, c: Mapping[LayerId, float], d: Mapping[LayerId, int], kappa: float = 1.0
) -> Dict[LayerId, float]:
    """A_ℓ ≈ (d_ℓ · c_ℓ^2) / 3 · κ を返す。
    - 量子化誤差の高レート近似 D_ℓ(B)=A_ℓ·2^{-2B} の係数近似。
    - κ は分布補正（外乱吸収）用のスカラー。
    """
    out: Dict[LayerId, float] = {}
    for lid, cval in c.items():
        out[lid] = (d.get(lid, 1) * float(cval) * float(cval)) / 3.0 * float(kappa)
    return out


class WaterfillBitAllocator(IBitAllocator):
    """C4–C5: 層別ビット割当（水充填→整数化）の参照実装（IBitAllocator 準拠）

    責務（Responsibilities）
    ------------------------
    - 連続緩和解（KKT, 水充填） B*_ℓ を返す: allocate(...)
    - 連続解を整数化して制約を満たす b_ℓ を返す: integerize(...)

    設計方針（抽象）
    ----------------
    - 連続解は λ を二分探索で求める（Σ d_ℓ·B*_ℓ = B_total）。
    - 整数化は floor(B*) から開始し、余り R を限界効用 ΔJ 最大へ貪欲配分。
      （d_ℓ 単位の消費に注意）

    パラメータ
    ----------
    Bmax : Optional[int]
        各層の上限ビット（None なら無制限）
    eps : float
        数値安定の微小値
    """

    def __init__(self, *, Bmax: Optional[int] = None, eps: float = 1e-12) -> None:
        self.Bmax = Bmax
        self.eps = float(eps)

    # ---- IBitAllocator.allocate ------------------------------------------------
    def allocate(
        self,
        W: Mapping[LayerId, float],
        A: Mapping[LayerId, float],
        B_total: int,
        d: Mapping[LayerId, int],
        Bmin: int = 2,
    ) -> Dict[LayerId, float]:
        """連続緩和（KKT;水充填）の解 B*_ℓ を返す。

        I/F（IBitAllocator）
        --------------------
        - 戻り値: {layer_id: float} 連続解（デバッグ/可視化/整数化の前段）

        制約
        ----
        - Σ d_ℓ·B*_ℓ = B_total（数値誤差内）
        - B*_ℓ ≥ Bmin, （Bmax があれば B*_ℓ ≤ Bmax）
        """
        lids: List[LayerId] = list(W.keys())
        if not lids:
            return {}

        # 非負化ガード
        eps = self.eps
        Wp = {lid: max(float(W[lid]), eps) for lid in lids}
        Ap = {lid: max(float(A[lid]), eps) for lid in lids}
        dp = {lid: int(d.get(lid, 1)) for lid in lids}

        # 端のチェック：下限合計
        min_need = sum(dp[lid] * int(Bmin) for lid in lids)
        if B_total <= min_need:
            return {lid: float(Bmin) for lid in lids}

        # 目的: Σ d_ℓ·max{Bmin, min(Bmax, 0.5 log2(WA/λ))} = B_total となる λ を求める
        def sum_bits_given_lambda(lmbd: float) -> float:
            total = 0.0
            for lid in lids:
                val = 0.5 * math.log2(Wp[lid] * Ap[lid] / max(lmbd, eps))
                if self.Bmax is not None:
                    val = min(val, float(self.Bmax))
                total += dp[lid] * max(float(Bmin), val)
            return total

        lo, hi = eps, max(Wp[lid] * Ap[lid] for lid in lids) * 2.0
        target = float(B_total)
        for _ in range(60):  # 高精度に十分
            mid = (lo + hi) / 2.0
            s = sum_bits_given_lambda(mid)
            if s > target:
                lo = mid
            else:
                hi = mid
        lam = (lo + hi) / 2.0

        # 連続解 B*
        B_star: Dict[LayerId, float] = {}
        for lid in lids:
            val = 0.5 * math.log2(Wp[lid] * Ap[lid] / max(lam, eps))
            if self.Bmax is not None:
                val = min(val, float(self.Bmax))
            B_star[lid] = max(float(Bmin), val)
        return B_star

    # ---- IBitAllocator.integerize ---------------------------------------------
    def integerize(
        self,
        *,
        B_star: Mapping[LayerId, float],
        B_total: int,
        d: Mapping[LayerId, int],
        W: Mapping[LayerId, float],
        A: Mapping[LayerId, float],
    ) -> Dict[LayerId, int]:
        """連続解 B*_ℓ を整数化して b_ℓ を返す。

        仕様
        ----
        - 初期: b_ℓ = floor(B*_ℓ)
        - 余り R = B_total - Σ d_ℓ·b_ℓ を、限界効用 ΔJ 最大層へ 1bit ずつ割当
          （d_ℓ 単位の減算に注意）
        """
        lids = list(B_star.keys())
        dp = {lid: int(d.get(lid, 1)) for lid in lids}

        b: Dict[LayerId, int] = {lid: int(math.floor(float(B_star[lid]))) for lid in lids}
        used = sum(dp[lid] * b[lid] for lid in lids)
        R = int(B_total - used)

        if R < 0:
            # 過剰使用 → Δ損失が最小の層から 1bit ずつ戻す
            R = -R
            while R > 0:
                lid = self._argmin_marginal_loss(b, W, A, lids)
                if b[lid] > 0:
                    b[lid] -= 1
                    R -= dp[lid]
                else:
                    # 全て 0 なら終了
                    break
            return b

        # 余りを ΔJ 最大の層へ
        while R > 0:
            lid = self._argmax_marginal_gain(b, W, A, lids)
            # Bmax があるなら尊重（到達済みなら次点を探す）
            if self.Bmax is not None and b[lid] >= int(self.Bmax):
                alt = self._argmax_marginal_gain(b, W, A, [x for x in lids if x != lid])
                if alt is None:
                    break
                lid = alt
            b[lid] += 1
            R -= dp[lid]
            if R < 0:
                b[lid] -= 1
                break
        return b

    # ---- 内部ユーティリティ ---------------------------------------------------
    @staticmethod
    def _deltaJ(W: float, A: float, b_now: int) -> float:
        """ΔJ(b)=W·[A·(2^{-2b}−2^{-2(b+1)})]（1bit 追加の利得）"""
        t0 = 2.0 ** (-2.0 * float(b_now))
        t1 = 2.0 ** (-2.0 * float(b_now + 1))
        return float(W) * float(A) * (t0 - t1)

    @classmethod
    def _argmax_marginal_gain(
        cls,
        b: Mapping[LayerId, int],
        W: Mapping[LayerId, float],
        A: Mapping[LayerId, float],
        lids: list,
    ) -> LayerId | None:
        best_lid = None
        best_gain = -1.0
        for lid in lids:
            gain = cls._deltaJ(float(W[lid]), float(A[lid]), int(b.get(lid, 0)))
            if gain > best_gain:
                best_gain = gain
                best_lid = lid
        return best_lid

    @classmethod
    def _argmin_marginal_loss(
        cls,
        b: Mapping[LayerId, int],
        W: Mapping[LayerId, float],
        A: Mapping[LayerId, float],
        lids: list,
    ) -> LayerId:
        # 1bit 減らした時の損失が最小の層
        best_lid = lids[0]
        best_loss = float("inf")
        for lid in lids:
            if int(b.get(lid, 0)) <= 0:
                continue
            loss = cls._deltaJ(float(W[lid]), float(A[lid]), int(b.get(lid, 0) - 1))
            if loss < best_loss:
                best_loss = loss
                best_lid = lid
        return best_lid
