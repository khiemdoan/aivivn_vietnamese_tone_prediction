from collections import Counter

import torch
from torch import nn, optim

from model import EncoderRNN, GreedySearchDecoder, LuongAttnDecoderRNN
from utils import device, n_grams
from vocab import EOS_token, PAD_token, SOS_token, UNK_token, Vocab

NGRAM = 4
attn_model = 'dot'
# attn_model = 'general'
# attn_model = 'concat'
embedding_dim = 64
hidden_size = 512
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 256
learning_rate = 0.0001
decoder_learning_ratio = 5.0


loadFilename = 'models/checkpoint.pt'
checkpoint = torch.load(loadFilename, map_location=device)

voc = Vocab()
voc.__dict__ = checkpoint['voc_dict']

embedding = nn.Embedding(voc.num_words, embedding_dim)
embedding.load_state_dict(checkpoint['embedding'])

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
encoder.load_state_dict(checkpoint['en'])

decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
decoder.load_state_dict(checkpoint['de'])

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
encoder_optimizer.load_state_dict(checkpoint['en_opt'])

decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
decoder_optimizer.load_state_dict(checkpoint['de_opt'])

# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder, voc)

PAD_index = voc.char2index(PAD_token)
SOS_index = voc.char2index(SOS_token)
EOS_index = voc.char2index(EOS_token)
UNK_index = voc.char2index(UNK_token)


def predict(sentence):
    ### Format input sentence as a batch
    # words -> indexes
    uppercase_map = [c.isupper() for c in sentence]
    sentence = sentence.lower()
    indexes_batch = [voc.sentence2indexes(sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.tensor(indexes_batch, dtype=torch.long).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    max_length = len(sentence)
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = []
    for i, token in enumerate(tokens):
        idx = token.item()
        if idx == SOS_index or idx == PAD_index:
            continue
        if idx == EOS_index:
            break
        if idx == UNK_index:
            decoded_words.append(sentence[i])
        else:
            decoded_words.append(voc.index2char(idx))
    for i in range(min(len(uppercase_map), len(decoded_words))):
        if uppercase_map[i] is True:
            decoded_words[i] = decoded_words[i].upper()
    return ''.join(decoded_words), scores


def predict_sentence(sentence):
    ngrams = list(n_grams(sentence, NGRAM))
    candidates = [Counter() for _ in range(len(ngrams) + NGRAM - 1)]

    for nid, ngram in enumerate(ngrams):
        words, scores = predict(ngram)
        for wid, word in enumerate(words.split()):
            candidates[nid + wid].update([word])
    candidates = [c for c in candidates if len(c)]
    return ' '.join(c.most_common(1)[0][0] for c in candidates)
