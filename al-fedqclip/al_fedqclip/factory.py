from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from al_fedqclip.compress.bitalloc_waterfill import WaterfillBitAllocator
from al_fedqclip.strategy import StrategyComponents
from al_fedqclip.compress.quant_clip import StochasticUniformQuantizer
from al_fedqclip.compress.clipper_mad_ema import ClipperMADEMA
from al_fedqclip.compress.server_clip_basic import BasicServerClipper
from al_fedqclip.compress.aggregator_layerwise import LayerwiseAggregator
from al_fedqclip.compress.importance_ema import ImportanceEMA


@dataclass
class FactoryConfig:
    gamma_e: float = 5.0 # C3 EF 安全上限
    eta_s:float = 1.0    # C7 基準スケール
    gamma_s: float = 1.0 # C7 クリップ強度
    Bmax: Optional[int] = None  # C4-C5 各層の上限ビット

class ComponentFactory:
    def __init__(self, cfg: Optional[FactoryConfig]=None) -> None:
        self.cfg = cfg or FactoryConfig()

    def build(self)->StrategyComponents:
        importance = ImportanceEMA()                                        # C1
        clipper = ClipperMADEMA()                                           # C2
        quantizer = StochasticUniformQuantizer(gamma_e=self.cfg.gamma_e)    # C3
        aggregator = LayerwiseAggregator()
        server_clipper = BasicServerClipper(eta_s=self.cfg.eta_s, gamma_s=self.cfg.gamma_s) # C7

        bit_allocator = WaterfillBitAllocator(Bmax=self.cfg.Bmax)

        return StrategyComponents(
            importance=importance,
            clipwidth=clipper,
            bit_allocator=bit_allocator,
            quantizer=quantizer,
            aggregator=aggregator,
            server_clipper=server_clipper,
            scheduler=None,
        )