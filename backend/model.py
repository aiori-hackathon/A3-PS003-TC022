
"""NoCivicSense: GRU traffic-flow prediction model."""

import torch
from torch import nn


class TrafficGRU(nn.Module):
    """Predict the next traffic-flow value from a sequence."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.gru(x)
        last_output = sequence_output[:, -1, :]
        return self.output(last_output)
