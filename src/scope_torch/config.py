from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence, Tuple

import torch


@dataclass(slots=True)
class SimulationConfig:
    """Shared simulation metadata and PyTorch runtime configuration."""

    roi_bounds: Tuple[float, float, float, float]
    start_time: datetime
    end_time: datetime
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    chunk_size: Optional[int] = 1024
    require_grad: bool = False
    extra_options: dict[str, float] = field(default_factory=dict)

    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def torch_dtype(self) -> torch.dtype:
        return self.dtype

    def chunks(self, total: int) -> Sequence[slice]:
        if not self.chunk_size or self.chunk_size <= 0:
            return [slice(0, total)]
        return [slice(i, min(i + self.chunk_size, total)) for i in range(0, total, self.chunk_size)]
