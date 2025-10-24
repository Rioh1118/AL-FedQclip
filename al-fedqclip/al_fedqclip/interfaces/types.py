# 共通の型・データ構造

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple
import torch

LayerId = int  # レイヤ番号(0, 1, 2)
ClientId = str # Flowerのクライアント識別子

try:
    Tensor = torch.Tensor
except Exception:
    Tensor = Any

@dataclass
class QuantizedPacket:
    """量子化済み1レイヤの送信パケット
    - q_bytes: 量子化後の整数配列バイト列 (ex: int16)
    - shape  : 復元用の元テンソル形状
    - B      : ビット幅(レベル数 = 2**B - 1)
    - c, m   : クリップ閾値と平均シフト
    - dtype  : 量子化配列のdtype("int16"など)
    """
    q_bytes: bytes
    shape: Tuple[int, ...]
    B: int
    c: float
    m: float
    dtype: str = "int16"

@dataclass
class LayerStats:
    """安全集約する層スカラー統計 (プライバシ配慮のため平均化スカラーのみ)"""
    fisher_ema: float
    snr_ema: float = 0.0
    htr_ema: float = 0.0

