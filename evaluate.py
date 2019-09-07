import re

import torch
from torch import nn, optim

from lang import EOS_token, PAD_token, SOS_token, Voc
from model import EncoderRNN, GreedySearchDecoder, LuongAttnDecoderRNN
from utils import device

voc = Voc('hihi')

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

loadFilename = 'models/save/500/checkpoint.tar'
checkpoint = torch.load(loadFilename)

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

MAX_LENGTH = 4

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    # Tách dấu câu nếu kí tự liền nhau
    s = re.sub(r"([.!?,\-\&\(\)\[\]])", r" \1 ", s)
    # Thay thế nhiều spaces bằng 1 space.
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


# Begin chatting (uncomment and run the following lin
evaluateInput(encoder, decoder, searcher, voc)
