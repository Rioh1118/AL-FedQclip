from typing import Mapping, Protocol
from .types import LayerId, Tensor

class IServerClipper(Protocol):
    """サーバ側の大域Clipped SGD (C7)"""
    def apply(self, *, global_grad: Mapping[LayerId, Tensor])->Mapping[LayerId, Tensor]:...
