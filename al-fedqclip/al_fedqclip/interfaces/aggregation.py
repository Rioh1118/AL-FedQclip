from typing import Mapping, Dict, Protocol
from .types import ClientId, LayerId, QuantizedPacket, Tensor

class ILayerAggregator(Protocol):
    """層ごとの集約(欠損層対応、重み付き)"""
    def aggregate(self, *, packets_by_client: Mapping[ClientId, Mapping[LayerId, QuantizedPacket]], client_weights: Mapping[ClientId, float])->Dict[LayerId, Tensor]:...