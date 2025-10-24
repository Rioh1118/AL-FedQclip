"""
進行率連動スケジューラ(C6)
===============================

目的
______
- 学習の進行率 p=t/T に応じて、**均等ビット配分**から**最適ビット配分**へ滑らかに遷移させる。
- 学習終盤で**後段層**へ +1~2bit のボーナスを与え、微調整能力を確保する


設計
_______________
- 進行率　p は[0,1] へクリップ。
- シグモイドη(p) = 1/(1+exp(-a(p-b)))で補間し、
    B_dyn[l] = round((1-η)*B + η*B_current[l])
    とする。ここでBはB_currentの算術平均
- `tail_layers` に含まれる層へは学習終盤ほど効くように +bonusを付与(ηを重みとして付与)
- 総ビットの厳密保存は本層では行わない(丸め誤差が小さいため)。厳密保存が必要ならビットアロケーた側の整数化で調整、またはpost-fixを別途用意する
    ビットアロケータ側の整数化で調整、またはpost-fixを別途用意
"""

from __future__ import annotations
from typing import Mapping, Dict, Iterable
import math

from ..interfaces import IBitScheduler, LayerId

class SigmoidScheduler(IBitScheduler):
    """進行率シグモイド・スケジューラ

    Parameters
    __________
    a: float
        シグモイドの傾き(大きいほど切り替えが急)
    b: float
        シグモイドの中心(p=bでη=0.5)
    tail_weight: float
        後段層ボーナスの強度係数(0~1程度を推奨)
    """
    def __init__(self, a: float=10.0, b: float=0.5, tail_weight: float=1.0)->None:
        self.a = float(a)
        self.b = float(b)
        self.tail_weight = float(tail_weight)

    def _sigma(self, p: float) -> float:
        # 数値安定のためpを[0,1]クリップ
        p = 0.0 if p < 0.0 else (1.0 if p>1.0 else p)
        return 1.0/(1.0+math.exp(-self.a*(p-self.b)))

    def schedule(self, *, B_current: Mapping[LayerId, int] | Mapping[LayerId, float], progress: float, tail_layers: Iterable[LayerId] = (), bonus: int = 1 )->Dict[LayerId, int]:
        """均等->最適のスムーズな繊維を行い、終盤は後段層へボーナスを付与

        Notes
        ______
        - B_currentは「整数化後」のビットでも「連続値」のビットでも良い
        - 返り値はintに丸める
        - ボーナスはηに比例して付与(序盤は0に近く、終盤はfullに)。
        """
        keys = list(B_current.keys())
        if not keys:
            return {}

        # 均等ベースライン
        mean_B = sum(float(B_current[k]) for k in keys) / len(keys)

        eta = self._sigma(progress)
        out: Dict[LayerId, int] = {}

        # 本体の補間
        for l in keys:
            b_now = float(B_current[l])
            b_dyn = (1.0 - eta) * mean_B + eta * b_now
            out[l] = int(round(b_dyn))

        if bonus != 0:
            scaled_bonus = int(round(self.tail_weight * eta * bonus))
            if scaled_bonus != 0:
                tail_set = set(tail_layers)
                for l in keys:
                    if l in tail_set:
                        out[l] = max(0, out[l] + scaled_bonus)

        return out

__all__ =["SigmoidScheduler"]