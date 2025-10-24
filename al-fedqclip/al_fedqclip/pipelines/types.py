from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Dict, Iterable, List, Optional
from ..interfaces import LayerId, ClientId, Tensor, QuantizedPacket, LayerStats

@dataclass(frozen=True)
class ServerRoundInputs:
    """サーバ側 1ラウンドの入力DTO
    - d_per_layer : d_ℓ
    - clip_proxy  : c_ℓ, tの近似(EMA由来)。 A_ℓ, tの構成に利用
    - progress    : p=t/T(0..1)
    - B_total     : 総ビット予算(bit/round)
    - Bmin        : レイヤ最小ビット
    - tail_layers : 後段層のボーナス対象
    - he_masks    : HE用 端末→連続層マスク
    """
    d_per_layer: Mapping[LayerId, int]
    clip_proxy: Mapping[LayerId, float]
    progress: float
    B_total: int
    Bmin: int = 2
    tail_layers: Iterable[LayerId] = ()
    he_masks: Optional[Mapping[ClientId, List[int]]] = None

@dataclass(frozen=True)
class ServerRoundOutputs:
    """サーバ側 1ラウンドの出力DTO"""
    B_dyn: Dict[LayerId, int]                 # 配布する層別ビット
    g_clipped: Dict[LayerId, Tensor]          # 集約後の大域更新(Clipped済み)
    stats_updated: Dict[LayerId, LayerStats]  # 重要度更新後の統計

@dataclass(frozen=True)
class ClientRoundInputs:
    """クライアント側 1ラウンドの入力DTO"""
    B_dyn: Mapping[LayerId, int]
    layer_ids: Iterable[LayerId]
    mask: Optional[Mapping[LayerId, int]] = None
    K_local: int = 1

@dataclass(frozen=True)
class ClientRoundOutputs:
    """クライアント側 1ラウンドの出力DTO"""
    packets: Dict[LayerId, QuantizedPacket]
    stats: Dict[LayerId, LayerStats]