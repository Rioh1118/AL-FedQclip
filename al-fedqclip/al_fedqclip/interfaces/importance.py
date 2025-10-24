from typing import Mapping, Dict, Protocol, runtime_checkable, Optional
from .types import LayerId, Tensor, LayerStats

@runtime_checkable
class IImportanceEstimator(Protocol):
    """層の重要度推定
    - 勾配/統計からw_ellを更新し、正規化重みを返す
    - 実装は Fisher/SNRのEMAを推奨、Hessianは任意
    """
    def update_from_grads(self, grads_per_layer: Mapping[LayerId, Tensor]) -> None: ...
    """各層の勾配/更新ベクトルから統計を更新する"""
    def update_from_states(self, stats_per_layer: Mapping[LayerId, LayerStats]) -> None: ...
    """安全集約などで得た層スカラー統計を取り込み、内部統計を更新する"""
    def get_normalized_weights(self, d: Optional[Mapping[LayerId,int]]=None, *, size_weighting: bool = False,) -> Dict[LayerId, float]: ...
    """正規化重みを返す
    - size_weighting=False: w_ℓを返す （純粋な重要度、Σw_ℓ=1）
    - size_weighting=True かつd提供: W_ℓ = d_ℓ ・ w_ℓを正規化して返す(ΣW_ℓ=1)
    """

