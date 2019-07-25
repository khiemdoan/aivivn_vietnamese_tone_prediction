import itertools
import random
import re
from zipfile import ZipFile

import torch
from torch import nn, optim
from tqdm import tqdm

from lang import EOS_token, Lang, SOS_token
from model import AttnDecoderRNN, EncoderRNN
from utils import device, save_model
from vietnamese_utils import remove_vietnamese_tone

print('Pytorch version:', torch.__version__)

MAX_LENGTH = 50
LEARNING_RATE = 0.01
EPOCHS = 1000
TEACHER_FORCING_RATIO = 0.5
HIDDEN_SIZE = 256


def extract_phrases(text):
    return re.findall(r'\w[\w ]+', text)


input_lang = Lang('No-tone Vietnamese')
output_lang = Lang('Toned Vietnamese')

print('Read data...')
# with open('data/vietnamese_tone_prediction.zip', 'rb') as infile:
#     with ZipFile(infile) as inzip:
#         lines = inzip.read('train.txt').decode('utf-8').split('\n')

with open('data/mini_train.txt', 'r', encoding='utf-8') as infile:
    lines = infile.read().split('\n')

lines = itertools.chain.from_iterable(extract_phrases(line) for line in lines)
lines = [line for line in lines if len(line.split(' ')) < MAX_LENGTH - 2]
lines = [line.lower() for line in lines]

pairs = [(remove_vietnamese_tone(line), line) for line in lines]

for src, dest in pairs:
    input_lang.add_sentence(src)
    output_lang.add_sentence(dest)

print('{}: {} words'.format(input_lang.name, input_lang.n_words))
print('{}: {} words'.format(output_lang.name, output_lang.n_words))


def tensors_from_pair(pair):
    input_tensor = input_lang.sentence2tensor(pair[0])
    target_tensor = output_lang.sentence2tensor(pair[1])
    return input_tensor, target_tensor


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[output_lang.word2index[SOS_token]]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)

        loss += criterion(decoder_output, target_tensor[di])

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            if decoder_input.item() == output_lang.word2index[EOS_token]:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
decoder = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)

print('Train...')
encoder_optimizer = optim.SGD(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()

training_pairs = [tensors_from_pair(pair) for pair in pairs]

best_loss = float('inf')

for epoch in range(EPOCHS):
    random.shuffle(training_pairs)

    total_loss = 0

    for input_tensor, target_tensor in tqdm(training_pairs, desc='Epoch {:04d}'.format(epoch + 1)):
        loss = train(input_tensor, target_tensor,
                     encoder, decoder,
                     encoder_optimizer, decoder_optimizer,
                     criterion, MAX_LENGTH)
        total_loss += loss
    avg_loss = total_loss / len(training_pairs)
    if avg_loss < best_loss:
        best_loss = avg_loss
        print('Loss:', best_loss)
        save_model(encoder, decoder, best_loss)
