from typing import Tuple, Optional, Protocol
from .types import Tensor, QuantizedPacket

class IQuantizer(Protocol):
    """確率一様量子化+EF(C3)
    - 不偏量子化とEF残差の安全上限を担保
    """
    def quantize(self, *, u: Tensor, c:float, B: int, e_prev: Tensor) -> Tuple[QuantizedPacket, Tensor]: ...
    def dequantize(self, packet: QuantizedPacket, device: Optional[str] = None) -> Tensor: ...
