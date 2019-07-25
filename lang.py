import torch

from utils import device
from vietnamese_utils import is_vietnamese_word

SOS_token = '<SOS>'
EOS_token = '<EOS>'
Unknown_token = '<UNKNOWN>'


class Lang:

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0

        self.add_char(SOS_token)
        self.add_char(EOS_token)
        self.add_char(Unknown_token)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.name

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            if is_vietnamese_word(word):
                self.add_char(word)
            else:
                self.add_char(Unknown_token)

    def add_char(self, char):
        if char not in self.word2index:
            self.word2index[char] = self.n_words
            self.index2word[self.n_words] = char
            self.n_words += 1

    def sentence2indexes(self, sentence):
        indexes = [self.word2index[SOS_token]]
        for word in sentence.split(' '):
            if is_vietnamese_word(word):
                indexes.append(self.word2index.get(word, self.word2index[Unknown_token]))
            else:
                indexes.append(self.word2index[Unknown_token])
        indexes.append(self.word2index[EOS_token])
        return indexes

    def indexes2sentence(self, indexes):
        chars = []
        for index in indexes:
            char = self.index2word.get(index, self.index2word[2])
            if char == SOS_token:
                continue
            if char == EOS_token:
                continue
            chars.append(char)
        return chars

    def sentence2tensor(self, sentence):
        indexes = self.sentence2indexes(sentence)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def tensor2sentence(self, tensor):
        indexes = tensor.view(1, -1)[0].tolist()
        return self.indexes2sentence(indexes)
