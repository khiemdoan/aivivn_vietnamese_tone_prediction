from typing import Tuple

import torch
from torch import nn, tensor
from torch.nn import functional as F

from utils import device


class EncoderRNN(nn.Module):

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input: tensor, hidden: tensor) -> Tuple[tensor, tensor]:
        batch_size = input.size(0)
        embedded = self.embedding(input).view(1, batch_size, self.hidden_size)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size: int) -> tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input: tensor, hidden: tensor) -> Tuple[tensor, tensor]:
        batch_size = input.size(0)
        embedded = self.embedding(input).view(1, batch_size, self.hidden_size)
        output, hidden = self.gru(embedded, hidden)
        output = F.relu(output)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

    def init_hidden(self, batch_size: int) -> tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
