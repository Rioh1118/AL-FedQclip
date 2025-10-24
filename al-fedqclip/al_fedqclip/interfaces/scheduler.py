from __future__ import annotations

from typing import Mapping, Iterable, Dict, Protocol
from .types import LayerId

class IBitScheduler(Protocol):
    """進行率連動スケジューラ(C6)"""
    def schedule(self, *, B_current: Mapping[LayerId, int] | Mapping[LayerId, float], progress: float, tail_layers: Iterable[LayerId] = (), bonus: int = 1 )->Dict[LayerId, int]:...
