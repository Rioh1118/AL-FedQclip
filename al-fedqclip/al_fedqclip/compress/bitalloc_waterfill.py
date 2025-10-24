"""
C4–C5: 層別ビット割当（水充填＋整数化）
====================================

目的
----
- 1ラウンド内で量子化誤差の高レート近似
    D_ℓ(B) = A_ℓ · 2^{-2B}
  を最小化しつつ、通信制約
    Σ_ℓ d_ℓ · B_ℓ = B_total
  を満たすビット配分 B_ℓ を求める。
- 連続緩和（KKT；水充填） → 整数化（限界効用の貪欲配分）。

入出力（簡約）
--------------
- 入力：
  - W: Dict[LayerId, float]
    重要度重み（正規化推奨）。ImportanceEMA_v2.get_normalized_weights(size_weighting=True, d=layer_sizes) を想定。
  - d: Dict[LayerId, int]
    各層の次元数（通信コストの係数）。
  - B_total: int
    ラウンド当たりの総ビット制約。
  - A: Optional[Dict[LayerId, float]]
    層ごとのノイズ係数（なければ c と d から計算）。
  - c: Optional[Dict[LayerId, float]]
    クリップ幅（Clipperの c_ℓ）。A_ℓ = (d_ℓ · c_ℓ^2) / 3 * kappa で近似。
  - kappa: float = 1.0
    分布補正係数（A の外乱を吸収するスカラー）。
  - Bmin: int = 2
    各層の下限ビット（不安定なら 3）。
  - Bmax: Optional[int]
    各層の上限ビット（None なら無制限）。

- 出力：
  - b: Dict[LayerId, int]
    整数ビット配分。
  - B_star: Dict[LayerId, float]
    連続緩和の解（参考）。

アルゴリズム
------------
1) 連続水充填：
   B*_ℓ = max{Bmin, 0.5·log2( (W_ℓ·A_ℓ) / λ )}
   λ は Σ d_ℓ·B*_ℓ = B_total を満たすよう二分探索で決定。
   （Bmax があれば min{·, Bmax} も入れる）

2) 整数化：
   b_ℓ = floor(B*_ℓ)
   余り R = B_total - Σ d_ℓ·b_ℓ を、限界効用
   ΔJ_ℓ(b) = W_ℓ·[D_ℓ(b) − D_ℓ(b+1)]
   の最大な層（1 bitあたりの改善が大きい層）に 1 bit ずつ配る。
   d_ℓ で重い層は 1 配分= d_ℓ bit 消費になる点に注意（R を d_ℓ 単位で減算）。

備考
----
- D(B) の形が A·2^{-2B} なので、平均ビット/要素が 1 増えると誤差は ≈ ×1/4。
- 数値安定のため、W_ℓ·A_ℓ が 0 に近い層は Bmin に固定。
"""
from __future__ import annotations
from typing import Dict, Mapping, Optional, Tuple
import math

from ..interfaces import LayerId


def compute_A_from_clip(
    *, c: Mapping[LayerId, float], d: Mapping[LayerId, int], kappa: float = 1.0
) -> Dict[LayerId, float]:
    """A_ℓ = (d_ℓ · c_ℓ^2) / 3 · κ を計算して返す。
    クリップ幅 c_ℓ と層次元 d_ℓ から量子化歪みの係数を近似。
    """
    out: Dict[LayerId, float] = {}
    for lid, cval in c.items():
        out[lid] = (d.get(lid, 1) * float(cval) * float(cval)) / 3.0 * float(kappa)
    return out


def allocate_bits_waterfill(
    *,
    W: Mapping[LayerId, float],
    d: Mapping[LayerId, int],
    B_total: int,
    A: Optional[Mapping[LayerId, float]] = None,
    c: Optional[Mapping[LayerId, float]] = None,
    kappa: float = 1.0,
    Bmin: int = 2,
    Bmax: Optional[int] = None,
    eps: float = 1e-12,
) -> Tuple[Dict[LayerId, int], Dict[LayerId, float]]:
    """水充填＋整数化による層別ビット配分を実行。

    Returns
    -------
    b : Dict[LayerId, int]
        整数ビット配分（通信実行用）。
    B_star : Dict[LayerId, float]
        連続緩和解（デバッグ/可視化用）。
    """
    if A is None:
        if c is None:
            raise ValueError("allocate_bits_waterfill: either A or c must be provided")
        A = compute_A_from_clip(c=c, d=d, kappa=kappa)

    lids = list(W.keys())
    if not lids:
        return {}, {}

    # 0/負値の除去とガード
    Wp = {lid: max(float(W[lid]), eps) for lid in lids}
    Ap = {lid: max(float(A.get(lid, 0.0)), eps) for lid in lids}
    dp = {lid: int(d.get(lid, 1)) for lid in lids}

    # ── 連続水充填：λ を二分探索で求める ───────────────────────────
    def sum_bits_given_lambda(lmbd: float) -> float:
        total = 0.0
        for lid in lids:
            val = 0.5 * math.log2(Wp[lid] * Ap[lid] / max(lmbd, eps))
            if Bmax is not None:
                val = min(val, float(Bmax))
            total += dp[lid] * max(float(Bmin), val)
        return total

    # λ の探索区間を粗く決める（増えると B 減る）
    # 下限: ほぼ 0 → B が大きい / 上限: W*A の最大程度
    lo, hi = eps, max(Wp[lid] * Ap[lid] for lid in lids) * 2.0

    # 端のチェック：B_total が小さすぎる/大きすぎる場合
    min_need = sum(dp[lid] * Bmin for lid in lids)
    if B_total <= min_need:
        # 予算が下限合計未満なら、下限で詰めて終了
        b0 = {lid: int(Bmin) for lid in lids}
        return b0, {lid: float(Bmin) for lid in lids}

    # 収束条件
    target = float(B_total)
    for _ in range(60):  # 高精度に 1e-9 近辺まで
        mid = (lo + hi) / 2.0
        s = sum_bits_given_lambda(mid)
        if s > target:
            # B が多い → λ を上げる
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2.0

    # 連続解 B*
    B_star: Dict[LayerId, float] = {}
    for lid in lids:
        val = 0.5 * math.log2(Wp[lid] * Ap[lid] / max(lam, eps))
        if Bmax is not None:
            val = min(val, float(Bmax))
        B_star[lid] = max(float(Bmin), val)

    # ── 整数化：床関数＋限界効用で余りを配分 ────────────────────────
    b: Dict[LayerId, int] = {lid: int(math.floor(B_star[lid])) for lid in lids}

    # 残り R = B_total - Σ d_l · b_l を配る（d_ℓ 単位）
    used = sum(dp[lid] * b[lid] for lid in lids)
    R = int(B_total - used)
    if R < 0:
        # 過剰に使ってしまった場合（丸め下限やBmaxで起こり得る）→大きい層から削る
        # ただし通常は起きにくい
        R = -R
        # ΔJ の小さい順（削減の損失が小さい）に1bitずつ戻す
        while R > 0:
            lid = _argmin_marginal_loss(b, Wp, Ap, lids)
            if b[lid] > Bmin:
                b[lid] -= 1
                R -= dp[lid]
            else:
                # 全てBminだと終了
                break
        return b, B_star

    # 余りを ΔJ 最大へ配分
    while R > 0:
        lid = _argmax_marginal_gain(b, Wp, Ap, lids)
        # Bmax があればガード
        if Bmax is not None and b[lid] >= Bmax:
            # 次点を探す（単純に一回スキップ）
            alt = _argmax_marginal_gain(b, Wp, Ap, [x for x in lids if x != lid])
            if alt is None:
                break
            lid = alt
        b[lid] += 1
        R -= dp[lid]
        if R < 0:
            # 使い過ぎた分は戻す
            b[lid] -= 1
            break

    return b, B_star


# ── ユーティリティ ─────────────────────────────────────────────────

def _DeltaJ(W: float, A: float, b_now: int) -> float:
    """限界効用 ΔJ(b) = W·[D(b) − D(b+1)] = W·A·(2^{-2b} − 2^{-2(b+1)})."""
    t0 = 2.0 ** (-2.0 * float(b_now))
    t1 = 2.0 ** (-2.0 * float(b_now + 1))
    return float(W) * float(A) * (t0 - t1)


def _argmax_marginal_gain(
    b: Mapping[LayerId, int], W: Mapping[LayerId, float], A: Mapping[LayerId, float], lids: list
) -> LayerId:
    best_lid = lids[0]
    best_gain = -1.0
    for lid in lids:
        gain = _DeltaJ(W[lid], A[lid], b.get(lid, 0))
        if gain > best_gain:
            best_gain = gain
            best_lid = lid
    return best_lid


def _argmin_marginal_loss(
    b: Mapping[LayerId, int], W: Mapping[LayerId, float], A: Mapping[LayerId, float], lids: list
) -> LayerId:
    # 1bit 減らしたときの損失が最も小さい層
    best_lid = lids[0]
    best_loss = float("inf")
    for lid in lids:
        if b.get(lid, 0) <= 0:
            continue
        loss = _DeltaJ(W[lid], A[lid], b.get(lid, 0) - 1)
        if loss < best_loss:
            best_loss = loss
            best_lid = lid
    return best_lid


__all__ = [
    "allocate_bits_waterfill",
    "compute_A_from_clip",
]
