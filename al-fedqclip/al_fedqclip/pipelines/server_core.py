from typing import Mapping
from ..interfaces import ILayerAggregator, IServerClipper, IQuantizer, ClientId, LayerId, QuantizedPacket, Tensor

class ServerCore:
    """* サーバ側"実行"を担当(計画はBitPlannerに移譲)
    責務:
      - 受信パケットの復元→層別集約
      - サーバ側Clippedを適用して大域更新を返す
    """
    def __init__(self, *, aggregator: ILayerAggregator, server_clipper: IServerClipper, quantizer: IQuantizer):
        self.agg = aggregator
        self.sclip = server_clipper
        self.quant = quantizer

    def aggregate_and_clip(self, *, packets_by_client: Mapping[ClientId, Mapping[LayerId, QuantizedPacket]],
                           client_weights: Mapping[ClientId, float], )->Mapping[LayerId, Tensor]:
        agg = self.agg.aggregate(packets_by_client=packets_by_client, client_weights=client_weights)
        g_clip = self.sclip.apply(global_grad=agg)
        return g_clip
