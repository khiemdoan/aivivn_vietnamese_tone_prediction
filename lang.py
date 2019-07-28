from typing import Dict, List

import torch

from utils import device
from vietnamese_utils import is_vietnamese_word

SOS_token = '<SOS>'
EOS_token = '<EOS>'
Unknown_token = '<UNKNOWN>'


class Lang:

    def __init__(self, name: str) -> None:
        self.name = name
        self.word2index: Dict[str, int] = {}
        self.index2word: Dict[int, str] = {}

        self.add_char(SOS_token)
        self.add_char(EOS_token)
        self.add_char(Unknown_token)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.name

    def add_sentence(self, sentence: str) -> None:
        for word in sentence.split(' '):
            if is_vietnamese_word(word):
                self.add_char(word)
            else:
                self.add_char(Unknown_token)

    def add_char(self, char: str) -> None:
        if char not in self.word2index:
            next_index = self.n_words
            self.word2index[char] = next_index
            self.index2word[next_index] = char

    @property
    def n_words(self) -> int:
        return len(self.word2index)

    def sentence2indexes(self, sentence: str) -> List[int]:
        indexes = [self.word2index[SOS_token]]
        for word in sentence.split(' '):
            index = self.word2index.get(word, self.word2index[Unknown_token])
            indexes.append(index)
        indexes.append(self.word2index[EOS_token])
        return indexes

    def indexes2sentence(self, indexes: List[int]) -> List[str]:
        chars = []
        for index in indexes:
            char = self.index2word.get(index, Unknown_token)
            chars.append(char)
        return chars

    def sentence2tensor(self, sentence: str) -> torch.tensor:
        indexes = self.sentence2indexes(sentence)
        return torch.tensor(indexes, dtype=torch.long, device=device)

    def tensor2sentence(self, tensor: torch.tensor) -> List[str]:
        indexes = tensor.tolist()
        return self.indexes2sentence(indexes)
