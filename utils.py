import pickle
import re
from pathlib import Path
from typing import List, NoReturn, Tuple

import torch
from IPython.display import HTML, display
from torch.nn import Module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = Path('./models')


def save_model(epoch: int, loss: float, voc_dict, embedding, en, de, en_opt, de_opt) -> NoReturn:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f'epoch_{epoch}_loss_{loss:.5f}.pt'
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'voc_dict': voc_dict,
        'embedding': embedding,
        'encoder': en,
        'decoder': de,
        'en_opt': en_opt,
        'de_opt': de_opt,
    }, model_file)


def load_model(epoch: int):
    model_file = model_dir / 'epoch_{:.5f}.pt'.format(epoch)
    return load_model_file(model_file)


def load_model_file(file_path):
    checkpoint = torch.load(file_path)
    return checkpoint['epoch'], checkpoint['loss'],\
           checkpoint['voc_dict'], checkpoint['embedding'],\
           checkpoint['encoder'], checkpoint['decoder'],\
           checkpoint['en_opt'], checkpoint['de_opt'],


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
