import pickle
import re
from pathlib import Path
from typing import List, NoReturn, Tuple

import torch
from IPython.display import display, HTML
from torch.nn import Module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = Path('./models')


def save_model(encoder: Module, decoder: Module, accurate: float, loss: float) -> NoReturn:
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
    return load_model_file(model_file)


def load_model_file(file_path) -> Tuple[Module, Module, float, float]:
    checkpoint = torch.load(file_path)
    return checkpoint['encoder'].to(device), checkpoint['decoder'].to(device),\
           checkpoint['accurate'], checkpoint['loss']


def extract_phrases(text) -> List[str]:
    return re.findall(r'\w[\w ]+', text)


def pickle_dump(obj, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as outfile:
        pickle.dump(obj, outfile)


def pickle_load(file_path):
    with open(file_path, 'rb') as infile:
        return pickle.load(infile)


def get_display(name: str):
    class Display:
        def __init__(self):
            self.name = name

        def update(self, content):
            print(f'{self.name}: {content}')

    display_obj = display(HTML(name), display_id=True)
    if not display_obj:
        display_obj = Display()
    return display_obj
