from pathlib import Path
from typing import Tuple

import torch
from torch.nn import Module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = Path('./models')


def save_model(encoder: Module, decoder: Module, accurate: float, loss: float) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / 'accurate_{:.5f}_loss_{:.5f}.pt'.format(accurate, loss)
    torch.save({
        'encoder': encoder,
        'decoder': decoder,
        'accurate': accurate,
        'loss': loss,
    }, model_file)


def load_model(accurate: float, loss: float) -> Tuple[Module, Module, float, float]:
    model_file = model_dir / 'accurate_{:.5f}_loss_{:.5f}.pt'.format(accurate, loss)
    checkpoint = torch.load(model_file)
    return checkpoint['encoder'].to(device), checkpoint['decoder'].to(device), accurate, loss
