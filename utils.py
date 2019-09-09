import math
import time
from pathlib import Path
from typing import Generator, Tuple

import nltk
import torch
from IPython.display import HTML, display

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = Path('./models')


def n_grams(text: str, length=4) -> Generator[str, None, None]:
    sequence = text.split()
    if len(sequence) < length:
        yield ' '.join(sequence)
    for ngram in nltk.ngrams(sequence, length):
        yield ' '.join(ngram)


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


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f'{int(m)}m {int(s)}s'


def time_since(since: float, percent: float) -> Tuple[float, float, float]:
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return s, es, rs
