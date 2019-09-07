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
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)
