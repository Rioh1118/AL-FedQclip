from typing import Mapping, Dict, Protocol
from .types import LayerId

class IBitAllocator(Protocol):
    """ビット割り当て(レート-歪み最適化) (C4-C5)
    - 連続緩和(KKT, 水充填) -> 整数化
    """
    def allocate(self, W: Mapping[LayerId, float],A: Mapping[LayerId, float], B_total: int, d: Mapping[LayerId, int], Bmin: int = 2) -> Dict[LayerId, float]: ...
    def integerize(self, *, B_star: Mapping[LayerId, float], B_total: int, d: Mapping[LayerId, int], W: Mapping[LayerId, float], A: Mapping[LayerId, float]) -> Dict[LayerId, int]: ...
