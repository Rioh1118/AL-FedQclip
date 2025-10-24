"""
Layerwise Aggregation: 復元 + 層ごと加重平均
===========================================

目的
----
各クライアントから受信した層別の量子化パケットを復元し、
クライアント重み `p_i`（データ割合など）で **層ごとに加重平均** する。

想定するパケット（`QuantizedPacket`）
-------------------------------------
- q:     整数量子化テンソル（int8/16/32）
- scale: 量子化ステップ Δ（float）
- mean:  平均シフト m（float）
- B:     使用ビット数（int） ※復元には直接不要
- shape: 元のテンソル形状（tuple）
- dtype: 元 dtype（str; 'torch.float32' 等）

I/O
---
aggregate(
    packets_by_client: {client_id: {layer_id: QuantizedPacket}},
    client_weights:    {client_id: float},
) -> {layer_id: Tensor}

挙動
----
- **層ごと**に、当該層を送ってきたクライアント集合だけで重みを正規化（部分参加対応）。
- 復元は `x̂ = mean + q * scale` → 形状と dtype を元に戻す。
- 欠損（層を送ってこない）クライアントはその層の平均から除外。
- 数値安定のため eps を用意。総重みが0に近い場合はゼロテンソルを返す。

注意
----
- **サーバ側Clipping**（global clipped SGD）は別コンポーネント (`server_clip_basic`) に委譲。
- ここでは純粋に復元と加重平均のみを担う。
"""
from __future__ import annotations
from typing import Dict, Mapping, Any, Tuple

import torch

from interfaces import LayerId, Tensor, QuantizedPacket
from .types import ClientId


class LayerwiseAggregator:
    def __init__(self, *, eps: float = 1e-12) -> None:
        self.eps = float(eps)

    @torch.no_grad()
    def aggregate(
        self,
        *,
        packets_by_client: Mapping[ClientId, Mapping[LayerId, QuantizedPacket]],
        client_weights: Mapping[ClientId, float],
    ) -> Dict[LayerId, Tensor]:
        # まず全層IDの集合を作る
        layer_ids = set()
        for cid, mp in packets_by_client.items():
            layer_ids.update(mp.keys())

        out: Dict[LayerId, Tensor] = {}
        for lid in layer_ids:
            # この層を送ってきたクライアントのみ集める
            xs = []
            ws = []
            last_meta: Tuple[Tuple[int, ...], torch.dtype, torch.device] | None = None

            for cid, pkts in packets_by_client.items():
                if lid not in pkts:
                    continue
                pkt = pkts[lid]
                x = self._decode(pkt)
                w = float(client_weights.get(cid, 0.0))
                if w <= 0.0:
                    continue
                # 形状・dtypeを記憶（最後に out を作るため）
                if last_meta is None:
                    last_meta = (tuple(x.shape), x.dtype, x.device)
                xs.append(x)
                ws.append(w)

            if not xs:
                # 誰も送ってこなければスキップ
                continue

            W = sum(ws)
            if W <= self.eps:
                # 重みがゼロならゼロテンソル
                shape, dtype, device = last_meta  # type: ignore
                out[lid] = torch.zeros(shape, dtype=dtype, device=device)
                continue

            # 正規化して加重平均
            acc = None
            for x, w in zip(xs, ws):
                coef = w / W
                acc = x * coef if acc is None else acc.add_(x, alpha=coef)
            out[lid] = acc  # type: ignore

        return out

    # ── 復元 ─────────────────────────────────────────────────────────
    def _decode(self, pkt: QuantizedPacket) -> Tensor:
        """QuantizedPacket から Tensor を復元する。
        NamedTuple/Dataclass/辞書の差異に耐性を持たせる。
        """
        # 汎用アクセス関数
        def get(field: str, default: Any = None) -> Any:
            if isinstance(pkt, dict):
                return pkt.get(field, default)
            # dataclass / namedtuple 互換
            return getattr(pkt, field, default)

        q = get("q")
        scale = float(get("scale"))
        mean = float(get("mean"))
        shape = tuple(get("shape"))
        dtype_str = str(get("dtype", "torch.float32"))

        # dtype 復元
        try:
            dtype = getattr(torch, dtype_str.split(".")[-1])
            if not isinstance(dtype, torch.dtype):
                dtype = torch.float32
        except Exception:
            dtype = torch.float32

        # q は CPU/GPU どちらでもOK。浮動に変換して復元
        x = q.to(dtype=torch.float32) * float(scale) + float(mean)
        x = x.view(*shape).to(dtype=dtype)
        return x


__all__ = ["LayerwiseAggregator"]
