"""
C3: 不偏量子化 + EF（誤差フィードバック） + クリップ
=================================================

目的
----
- 層差分 `u` に EF を加えた残差 `r = u + e_prev` を、対称クリップ幅 `c` で
  クリップして **不偏な確率的一様量子化** を行い、通信パケットを生成する。
- 復元側では `m + q * Δ`（平均シフト + ステップ）で再構成する。

量子化ルール（各層）
--------------------
- クリップ： `r_clip = clamp(r - m, -c, +c)`（`m` は層平均の送信用シフト）
- ステップ： `Δ = 2c / (2^B - 1)`
- レベル：  `q ∈ { -L, ..., 0, ..., +L }`, `L = 2^{B-1}-1`（対称中点量子化）
- **確率丸め**（不偏）：`y = r_clip/Δ` を下側 `k=floor(y)` とし、
  `q = k + Bernoulli(p)`, `p=y-k`。その後 `[−L, +L]` にクリップ。
- 復元：`x̂ = m + q*Δ`
- EF更新：`e_next = r - x̂`（クリップ済みで上限を持つ）
- EF安全上限：`||e_next||² ≤ γ_e ||u||²` ならOK。超えたら等方スケールで戻す。

I/O（想定インターフェース）
---------------------------
- 入力:  `u: Tensor`, `c: float`, `B: int`, `e_prev: Tensor`
- 出力:  `(pkt: QuantizedPacket, e_next: Tensor)`
  - `QuantizedPacket` は `types` 側のデータ構造を想定（q, scale, mean, B, shape, dtype 等）。
  - もしフィールド名が異なる場合でも、最低限 `q, scale, mean, B, shape, dtype` を持つ。

注意
----
- 実装は PyTorch テンソル上でベクトル化。`q` は `int8/16/32` を状況で選択。
- 乱数は `torch.rand`（デバイスに合わせる）。再現性が必要なら外側でseed制御。
- `mean` と `scale(=Δ)` は `float32` でパケットに格納（16bit固定小数にする場合は別途）
"""
from __future__ import annotations
from dataclasses import asdict
from typing import Mapping, Tuple, Any

import torch

from ..interfaces import IQuantizer, LayerId, Tensor, QuantizedPacket


class StochasticUniformQuantizer(IQuantizer):
    """確率的一様量子化（不偏）+ 平均シフト + EF + クリップ。

    Parameters
    ----------
    gamma_e : float
        EF 残差の安全上限係数（ ||e_next||² ≤ γ_e ||u||² ）。
    eps : float
        数値安定の微小値。
    keep_stats : bool
        デバッグ用に内部統計を保持するか。
    """

    def __init__(self, *, gamma_e: float = 5.0, eps: float = 1e-12, keep_stats: bool = False) -> None:
        self.gamma_e = float(gamma_e)
        self.eps = float(eps)
        self.keep_stats = bool(keep_stats)
        self._last_info: Mapping[str, Any] | None = None

    @torch.no_grad()
    def quantize(self, *, u: Tensor, c: float, B: int, e_prev: Tensor) -> Tuple[QuantizedPacket, Tensor]:
        """量子化してパケット化し、次ラウンドの EF を返す。

        - r = u + e_prev
        - m = mean(r)
        - r_c = clamp(r - m, -c, +c)
        - Δ = 2c / (2^B - 1)
        - q: 確率丸めで不偏化
        - x̂ = m + q*Δ
        - e_next = r - x̂ （安全上限で縮小）
        """
        assert B >= 1, "B (bits) must be >= 1"

        # 入力整形
        dev = u.device
        dtype = u.dtype
        r = (u + e_prev).detach()
        shape = u.shape

        # 平均シフト & クリップ（中心化して対称クリップ）
        m = torch.mean(r).to(torch.float32)
        r_center = (r - m).to(torch.float32)
        c_f = float(c)
        r_clip = torch.clamp(r_center, -c_f, +c_f)

        # 量子化ステップ Δ と レベル境界
        # L = 2^{B-1} - 1  （対称）
        L = int(2 ** (max(B, 1) - 1) - 1)
        if L < 1:
            # B=1 のとき L=0 にならないよう B>=2 を推奨（Bmin=2）
            L = 1
        denom = float(2 ** B - 1)
        delta = (2.0 * c_f) / denom

        # 確率丸め（不偏）： y = r_clip/Δ = k + frac
        y = r_clip / (delta + self.eps)
        k = torch.floor(y)
        frac = (y - k).clamp_(0.0, 1.0)  # 数値誤差対策
        # Bernoulli(frac) を同形に生成
        rnd = torch.rand_like(frac)
        q = k + (rnd < frac).to(k.dtype)
        q = q.clamp_(-L, +L)

        # 復元値と EF 残差
        xhat = m + q * delta
        e_next = (r_center - q * delta).to(dtype)
        # e_next は中心化後の残差なので、平均シフトを戻したい場合は u 側の次回 r=u+e_prev で再中心化される

        # 安全上限（等方スケーリング）
        e2 = float(torch.sum(e_next.to(torch.float64) ** 2).item())
        u2 = float(torch.sum(u.detach().to(torch.float64) ** 2).item()) + self.eps
        if e2 > self.gamma_e * u2:
            scale = (self.gamma_e * u2 / (e2 + self.eps)) ** 0.5
            e_next.mul_(scale)

        # 量子化テンソルの dtype 選択
        if B <= 8:
            q_dtype = torch.int8
        elif B <= 16:
            q_dtype = torch.int16
        else:
            q_dtype = torch.int32
        q = q.to(q_dtype)

        # パケット生成：QuantizedPacket の構造に合わせる
        # 想定フィールド: q, scale(Δ), mean(m), B, shape, dtype
        try:
            pkt = QuantizedPacket(
                q=q,  # 整数テンソル（CPU/デバイスは問わない）
                scale=float(delta),
                mean=float(m.item()),
                B=int(B),
                shape=tuple(shape),
                dtype=str(dtype),
            )
        except Exception:
            # NamedTuple/Dataclass の差異に耐えるためのフォールバック
            fields = {"q": q, "scale": float(delta), "mean": float(m.item()), "B": int(B),
                      "shape": tuple(shape), "dtype": str(dtype)}
            # 可能ならコンストラクタに辞書で渡す
            try:
                pkt = QuantizedPacket(**fields)  # type: ignore
            except Exception:
                # 最後の保険：dictを返す（呼び出し側が受け入れる前提）
                pkt = fields  # type: ignore

        if self.keep_stats:
            self._last_info = {
                "delta": float(delta),
                "mean": float(m.item()),
                "L": int(L),
                "u2": u2,
                "e2": e2,
            }

        return pkt, e_next


__all__ = ["StochasticUniformQuantizer"]
