from pathlib import Path

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = Path('./models')


def save_model(encoder, decoder, loss):
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / '{:.5f}.pt'.format(loss)
    torch.save({
        'encoder': encoder,
        'decoder': decoder,
        'loss': loss,
    }, model_file)


def load_model(loss):
    model_file = model_dir / '{:.5f}.pt'.format(loss)
    checkpoint = torch.load(model_file)
    return checkpoint['encoder'].to(device), checkpoint['decoder'].to(device), loss
