from typing import Dict, List, NoReturn

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

    def sentence2indexes(self, sentence) -> List[int]:
        return [self.char2index(char) for char in sentence] + [self.char2index(EOS_token)]
