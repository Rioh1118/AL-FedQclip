from typing import Dict
from ..interfaces import IImportanceEstimator, IBitAllocator, IBitScheduler, LayerId
from .types import ServerRoundInputs

class BitPlanner:
    """*サーバ側の"計画"のみを担当
    数式対応:
        - W_ℓ, t = d_ℓ * w_ℓ,t
        - A_ℓ, t = d_ℓ * c_ℓ, t^2 / 3
        - 連続解(KKT) -> 整数化 -> 進行率スケジュール
    """
    def __init__(self, *, importance: IImportanceEstimator, allocator: IBitAllocator,scheduler: IBitScheduler):
        self.imp = importance   # C1 重要度推定
        self.alloc = allocator  # C4-C5 ビット割り当て
        self.sched = scheduler  # C6 進行率スケジューラ

    def plan(self, inp: ServerRoundInputs) -> Dict[LayerId, int]:
        # 1. W_ℓ,t を作る（d_ℓ * w_ℓ,t）
        # 2. A_ℓ,t を作る（d_ℓ * c_ℓ,t^2 / 3）
        # 3. allocator.allocate(...) → 連続解 B_star
        # 4. allocator.integerize(...) → 整数化 B_int
        # 5. scheduler.schedule(...) → B_dyn（進行率で補正 & tail層ボーナス）
        w = self.imp.get_normalized_weights()
        W = {l: inp.d_per_layer[l] * w.get(l,0.0) for l in inp.d_per_layer}
        A = {l: inp.d_per_layer[l] * (inp.clip_proxy[l]**2) / 3.0 for l in inp.d_per_layer}
        B_star = self.alloc.allocate(W=W, A=A, B_total=inp.B_total, d=inp.d_per_layer, Bmin=inp.Bmin)
        B_int = self.alloc.integerize(B_star=B_star, B_total=inp.B_total, d=inp.d_per_layer, W=W, A=A)
        B_dyn = self.sched.schedule(B_current=B_int, progress=inp.progress, tail_layers=inp.tail_layers, bonus=1)
        return B_dyn

