import re

import torch
from torch import nn, optim

from model import EncoderRNN, GreedySearchDecoder, LuongAttnDecoderRNN
from utils import device
from vocab import EOS_token, PAD_token, SOS_token, UNK_token, Vocab

voc = Vocab()

attn_model = 'dot'
# attn_model = 'general'
# attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 100
learning_rate = 0.0001
decoder_learning_ratio = 5.0

loadFilename = 'models/checkpoint.tar'
checkpoint = torch.load(loadFilename, map_location=device)

voc.__dict__ = checkpoint['voc_dict']

embedding = nn.Embedding(voc.num_words, hidden_size)
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
searcher = GreedySearchDecoder(encoder, decoder)


def evaluate(searcher, voc, sentence):
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
    # indexes ->
    decoded_words = []
    for i, token in enumerate(tokens):
        idx = token.item()
        if idx == voc.char2index(UNK_token):
            decoded_words.append(sentence[i])
        else:
            decoded_words.append(voc.index2char(idx))
    for i in range(min(len(uppercase_map), len(decoded_words))):
        if uppercase_map[i] is True:
            decoded_words[i] = decoded_words[i].upper()
    return decoded_words


def evaluateInput(searcher, voc):
    while True:
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Evaluate sentence
            output_words = evaluate(searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ''.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


# Begin chatting (uncomment and run the following lin
evaluateInput(searcher, voc)
