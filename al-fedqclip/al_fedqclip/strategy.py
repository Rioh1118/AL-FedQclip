from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch

from al_fedqclip.compress.bitalloc_waterfill import compute_A_from_clip
from al_fedqclip.interfaces import IQuantizer, IBitAllocator, IClipper, IServerClipper, LayerId, Tensor, \
    QuantizedPacket, ClientId, IImportanceEstimator, ILayerAggregator, IBitScheduler
from al_fedqclip.compress.aggregator_layerwise import LayerwiseAggregator

# ------------------------------------------------------------------
# 目的
# -------------------------------------------------------------------
# - AL-FedQClipの1ラウンドの流れ(Client/Server双方)を「部品の組み合わせ」として
# 抽象的に定義し、各コンポーネントの差し替え・再構成を容易にする
# - 何をいつ呼ぶかという制御フローが責任であり、数式やアルゴリズムは各コンポネント(C1~C7)に移譲する
# ---------------------------------------------------------------------

@dataclass
class StrategyComponents:
    """AL-FedQClipを構成するコンポネント束

    必須:
        - importance (C1):
            weights(grads: Mapping[LayerId, Tensor]) -> Dict[LayerId, float]
            層重要度 W_ℓを返す(Fisher/SNR/Hessinanの軽量統合)
        - clipwidth (C2):
            estimate(grads: Mapping[LayerId, Tensor])->Dict[LayerId, float]
            層のクリップ幅を返す
        - bit_allocator(C4-C5):
            allocate(W, d, B_total, A=None, c=None, ...) ->
                (b: Dict[LayerId, int], B_star: Dict[LayerId, float]))
            KKT(水充填)->整数化で層別ビット b_ℓを返す
        - quantizer(C3):
            quantize(u: Tensor, c: float, B:int, e_prev: Tensor) ->
                (pkt: QuantizedPacket, e_next: Tensor)
            不偏一様量子化 + EF(安全上限) + クリップ、通信パケット化
        - aggregator:
            aggregate(packets_by_client, client_weights) -> Dict[LayerId, Tensor]
            層ごとの復元 + 加重平均
        - server_clipper: C7 サーバ側 Clipped更新
            apply(global_grad: Mapping[LayerId, Tensor]) -> Mapping[LayerId, Tensor]
            サーバ側の大域 Clipped 更新

    任意:
        scheduler: C6 スケジューラ(進行率に応じた補間/上書き等)
    """
    importance: IImportanceEstimator
    clipwidth: IClipper
    bit_allocator: IBitAllocator
    quantizer: IQuantizer
    aggregator: ILayerAggregator
    server_clipper: IServerClipper
    server_clipper: IServerClipper
    scheduler: Optional[IBitScheduler] = None

class ALFedQClipStrategy:
    """
    AL-FedQClipの抽象オーケストレータ
    前提（Assumptions）
    -------------------
    - QuantizedPacket は少なくとも {q, scale(Δ), mean(m), B, shape, dtype} を持つ。
    - EF バッファやクライアント状態の保存/読み出しは上位（ClientCore 等）が行う。

    不変条件（Invariants）
    ----------------------
    - Σ_ℓ d_ℓ · b_ℓ == B_total（bit_allocator の整数化結果）
    - 量子化（C3）は不偏性と EF 安全上限（||e_next||² ≤ γ_e||u||²）を満たす
    - 集約は「その層を送ったクライアントだけ」で重み正規化（部分参加対応）
    - サーバ Clipping（C7）は仕様の H_t を満たす
    """
    def __init__(self, comps: StrategyComponents):
        self.c = comps

    # ________________________________________
    # Client-side(各クライアントiの1ラウンド)
    # _______________________________________
    def client_round(
            self,
            *,
            client_id: ClientId,
            layer_updates: Mapping[LayerId, Tensor],    # u_{i, ℓ}
            e_prev: Mapping[LayerId, Tensor],           # EFバッファ(前ラウンド)
            layer_sizes: Mapping[LayerId, int],         # d_ℓ
            bit_budget: int,                            # B_total (i)も可
            client_weight: float,                       # p_i (サーバ加重平均用)
            progress: Optional[float] = None            # 進行率 p=t/T (C6)
    )->Dict[LayerId, QuantizedPacket]:
        """
        1. C1: importance.update_from_grads(u) -> get_normalized_weights(d, size_weighting=True) でW_ℓ
        2. C2: 各層に対してclipper.observe(ℓ, u_ℓ) → c_ℓ = clipper.get(ℓ)
        3. C4-C5: allocate(W, d, B_total, c)でb_ℓ
        4. C3: 各層 quantize(u_ℓ, c_ℓ, B=b_ℓ, e_prev_ℓ)→packet
        """
        # ------ C1: 層重要度 W_ℓ ----------------------------------------------------
        self.c.importance.update_from_grads(layer_updates)
        # W_ℓ = d_ℓ・w_ℓを正規化 (ΣW_ℓ=1)→bit allocatorへ渡す想定
        W = self.c.importance.get_normalized_weights(d=layer_sizes, size_weighting=True)

        # ------- C2: クリップ幅 c_ℓ ---------------------------------------------------
        c: Dict[LayerId, float] = {}
        for lid, u in layer_updates.items():
            # 観測: 直近の更新ベクトルから統計を更新
            self.c.clipwidth.observe(lid, u)
            # 取得: 現在のクリップ幅c_ℓを問い合わせ
            c[lid] = float(self.c.clipwidth.get(lid))

        #---- A係数を作成 (A_ℓ ≒ d_ℓ c_ℓ^2 / 3*k)
        A = compute_A_from_clip(c=c, d=layer_sizes,kappa=1.0)

        # -------C4-C5: ビット割り当て(水充填→整数化) ----------------------------------------
        B_star = self.c.bit_allocator.allocate(
            W=W, d=layer_sizes, B_total=int(bit_budget),  Bmin=2
        )
        b = self.c.bit_allocator.integerize(B_star=B_star, B_total=int(bit_budget), d=layer_sizes, W=W, A=A)

        # -----C6: 進行率スケジューリング ---------------------------------------------
        if self.c.scheduler is not None:
            b = self.c.scheduler.schedule(
                B_current=b, progress=float(progress), tail_layers=[], bonus=1
            )

        # -------C3: 量子化 + EF(層ごと) ---------------------------------------------
        pkts: Dict[LayerId, QuantizedPacket] = {}
        for lid, u in layer_updates.items():
            pkt, e_next = self.c.quantizer.quantize(u=u, c=float(c[lid]), B=int(b[lid]), e_prev=e_prev[lid])
            pkts[lid] = pkt
            # NOTE: e_nextの保存は上位(ClientCoreなど)に移譲

        return pkts

    def server_round(self,
                     *,
                     packets_by_client: Mapping[ClientId, Mapping[LayerId, QuantizedPacket]],
                     client_weights:Mapping[ClientId, float],)-> Mapping[LayerId, Tensor]:
        agg = self.c.aggregator.aggregate(
            packets_by_client=packets_by_client, client_weights=client_weights
        )
        for lid, x in agg.items():
            if not torch.isfinite(x).all():
                raise FloatingPointError(f"None-finite at layer {lid}")
        return self.c.server_clipper.apply(global_grad=agg)
