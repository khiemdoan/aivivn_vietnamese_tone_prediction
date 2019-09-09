from typing import Dict, List, NoReturn

import torch

from utils import device

# SOS_token = '<SOS>'
# EOS_token = '<EOS>'
# Unknown_token = '<UNKNOWN>'
#
#
# class Lang:
#
#     def __init__(self, name: str) -> NoReturn:
#         self.name = name
#         self.char2index: Dict[str, int] = {}
#         self.index2char: Dict[int, str] = {}
#
#         self.add_char(SOS_token)
#         self.add_char(EOS_token)
#         self.add_char(Unknown_token)
#
#     def __str__(self):
#         return self.__repr__()
#
#     def __repr__(self):
#         return self.name
#
#     def add_sentence(self, sentence: str) -> NoReturn:
#         for char in sentence:
#             self.add_char(char)
#
#     def add_char(self, char: str) -> NoReturn:
#         if char not in self.char2index:
#             next_index = self.n_words
#             self.char2index[char] = next_index
#             self.index2char[next_index] = char
#
#     @property
#     def n_words(self) -> int:
#         return len(self.char2index)
#
#     def sentence2indexes(self, sentence: str) -> List[int]:
#         indexes = [self.char2index[SOS_token]]
#         for char in sentence:
#             index = self.char2index.get(char, self.char2index[Unknown_token])
#             indexes.append(index)
#         indexes.append(self.char2index[EOS_token])
#         return indexes
#
#     def indexes2sentence(self, indexes: List[int]) -> str:
#         chars = []
#         for index in indexes:
#             if index == self.char2index[SOS_token]:
#                 continue
#             if index == self.char2index[EOS_token]:
#                 break
#             char = self.index2char.get(index, Unknown_token)
#             chars.append(char)
#         return ''.join(chars)
#
#     def sentence2tensor(self, sentence: str) -> torch.tensor:
#         indexes = self.sentence2indexes(sentence)
#         return torch.tensor(indexes, dtype=torch.long, device=device)
#
#     def tensor2sentence(self, tensor: torch.tensor) -> List[str]:
#         indexes = tensor.tolist()
#         return self.indexes2sentence(indexes)

# Default word tokens
PAD_token = '<PAD>'         # Used for padding short sentences
SOS_token = '<SOS>'         # Start-of-sentence token
EOS_token = '<EOS>'         # End-of-sentence token
UNK_token = '<UNKNOWN>'     # Unknown token


class Vocab:

    def __init__(self):
        self._char2index: Dict[str, int] = {}
        self._index2char: Dict[int, str] = {}

        self.add_char(PAD_token)
        self.add_char(SOS_token)
        self.add_char(EOS_token)
        self.add_char(UNK_token)

    def add_sentence(self, sentence: str) -> NoReturn:
        for char in sentence:
            self.add_char(char)

    def add_char(self, char: str) -> NoReturn:
        if char not in self._char2index:
            next_index = self.num_words
            self._char2index[char] = next_index
            self._index2char[next_index] = char

    @property
    def num_words(self) -> int:
        return len(self._char2index)

    def char2index(self, char: str) -> int:
        return self._char2index.get(char, self._char2index[UNK_token])

    def index2char(self, index: int) -> str:
        return self._index2char.get(index, UNK_token)

    def sentence2indexes(self, sentence):
        return [self.char2index(char) for char in sentence] + [self.char2index(EOS_token)]
