from ..interfaces import IClipper, IQuantizer, LayerId, Tensor, LayerStats, QuantizedPacket
from .types import ClientRoundInputs, ClientRoundOutputs
from typing import Mapping, Dict

class ClientCore:
    """* クライアント側"実行"を担当(学習本体は別レイヤ)
    責務:
        - 差分 u_ℓ, tに対してクリップ更新->不偏量子化+EF でパケット化
        - HE: maskに従って送信層のみ処理
    """
    def __init__(self, *, clipper: IClipper, quantizer: IQuantizer):
        self.clipper = clipper
        self.quant = quantizer
        self._ef: Dict[LayerId, Tensor] = {}

    def encode_updates(self, *, layer_updates: Mapping[LayerId, Tensor], inp: ClientRoundInputs) -> ClientRoundOutputs:
        packets: Dict[LayerId, QuantizedPacket] = {}
        stats: Dict[LayerId, LayerStats] = {}
        for l in inp.layer_ids:
            if inp.mask is not None and inp.mask.get(l, 1) == 0:
                continue
            u = layer_updates[l]
            self.clipper.observe(layer=l, update_tensor=u)
            c = self.clipper.get(layer=l)
            B = int(inp.B_dyn.get(l, 2))
            if l not in self._ef:
                self._ef[l] = u.detach().clone().zero_()
            pkt, e_next = self.quant.quantize(u=u, c=c, B=B, e_prev=self._ef[l])
            self._ef[l] = e_next.detach()
            packets[l] = pkt
            stats[l] = LayerStats(fisher_ema=float(u.detach().pow(2).mean().item()))
        return ClientRoundOutputs(packets=packets, stats=stats)
