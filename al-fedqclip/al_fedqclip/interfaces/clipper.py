from typing import Protocol
from .types import LayerId, Tensor

class IClipper(Protocol):
    """クリップ閾値管理(C2)
    - 初期化: MADベース推奨
    - EMA更新と95%分位バックストップ
    """
    def get(self, layer: LayerId)->float: ...
    def observe(self, layer: LayerId, update_tensor: Tensor)->None: ...
