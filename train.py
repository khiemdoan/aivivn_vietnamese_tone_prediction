import gc
import itertools
import platform
import random
import string
import sys
from zipfile import ZipFile

import torch
from progressbar import progressbar
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from lang import EOS_token, Lang, SOS_token
from model import DecoderRNN, EncoderRNN, SentenceDataSet
from utils import device, extract_phrases, get_display, pickle_dump, save_model
from vietnamese_utils import remove_vietnamese_tone, uni_chars_l

print(f'Python version: {platform.python_version()}')
print(f'Pytorch version: {torch.__version__}\n')

MAX_LENGTH = 50
LEARNING_RATE = 0.01
HIDDEN_SIZE = 256
TEACHER_FORCING_RATIO = 0.5
BATCH_SIZE = 1024
EPOCHS = 1000

if 'google.colab' not in sys.modules:
    BATCH_SIZE = 128

input_vocab = ' ' + string.ascii_lowercase
target_vocab = input_vocab + uni_chars_l

input_lang = Lang('No-tone Vietnamese')
target_lang = Lang('Toned Vietnamese')

input_lang.add_sentence(input_vocab)
target_lang.add_sentence(target_vocab)

print('Read data...\n')
with ZipFile('data/vietnamese_tone_prediction.zip', 'r') as inzip:
    lines = inzip.read('train.txt').decode('utf-8').split('\n')
    lines = lines[:1_000]

print('Preprocess...\n')
lines = itertools.chain.from_iterable(extract_phrases(line) for line in lines)
lines = [line for line in lines if len(line.split(' ')) < MAX_LENGTH - 2]
lines = [line.lower() for line in lines]
pairs = [(remove_vietnamese_tone(line), line) for line in lines]
del lines

print('Total sentences:', len(pairs))

print(f'{input_lang.name}: {input_lang.n_words} words')
print(f'{target_lang.name}: {target_lang.n_words} words')

pickle_dump(input_lang, 'data/input_lang.pickle')
pickle_dump(target_lang, 'data/target_lang.pickle')


def tensors_from_pair(pair):
    input_tensor = input_lang.sentence2tensor(pair[0]).to(device)
    input_tensor = F.pad(input_tensor, [0, MAX_LENGTH - input_tensor.size(0)],
                         'constant', input_lang.char2index[EOS_token])
    target_tensor = target_lang.sentence2tensor(pair[1]).to(device)
    target_tensor = F.pad(target_tensor, [0, MAX_LENGTH - target_tensor.size(0)],
                          'constant', target_lang.char2index[EOS_token])
    return input_tensor, target_tensor



print('Conv data')
pairs = [tensors_from_pair(pair) for pair in progressbar(pairs)]
training_pairs, validation_pairs = train_test_split(pairs, test_size=0.2)

training_dataset = SentenceDataSet(training_pairs)
training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)

validation_dataset = SentenceDataSet(validation_pairs)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

print()
print(f'Total trains: {len(training_pairs)}')
print(f'Total validations: {len(validation_pairs)}')
del pairs, training_pairs, training_dataset, validation_pairs, validation_dataset


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    input_tensor = input_tensor.transpose(0, 1)
    target_tensor = target_tensor.transpose(0, 1)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    batch_size = input_tensor.size(1)

    encoder_hidden = encoder.init_hidden(batch_size)
    encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = target_tensor[0]
    decoder_hidden = encoder_hidden

    use_teacher_forcing = random.random() < TEACHER_FORCING_RATIO

    for di in range(1, target_length):
        target_output = target_tensor[di]
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            loss += criterion(decoder_output, target_output)
            decoder_input = target_output   # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            topv, topi = decoder_output.topk(1, dim=1)
            decoder_input = topi.detach().transpose(1, 0)[0]  # detach from history as input
            loss += criterion(decoder_output, target_output)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def calc_accurate(input_tensor, target_tensor, encoder, decoder):
    input_tensor = input_tensor.transpose(0, 1)

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(1)

    batch_size = input_tensor.size(1)

    encoder_hidden = encoder.init_hidden(batch_size)
    encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = target_tensor[:, 0]
    decoder_hidden = encoder_hidden

    output = torch.zeros(target_length, batch_size, dtype=torch.int64, device=device)
    output[0] = decoder_input

    for di in range(1, target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1, dim=1)
        decoder_input = topi.detach().transpose(1, 0)[0]
        output[di] = decoder_input

    total = 0
    correct = 0

    for bi in range(batch_size):
        # print(target_lang.tensor2sentence(target_tensor[bi]))
        # print(target_lang.tensor2sentence(output[:, bi]))

        for wi in range(target_length):
            if output[wi][bi] == target_lang.char2index[EOS_token]:
                break

            total += 1
            if output[wi][bi] != target_tensor[bi][wi]:
                correct += 1

    return correct / total


encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
decoder = DecoderRNN(HIDDEN_SIZE, target_lang.n_words, MAX_LENGTH).to(device)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()

best_accurate = 0

print('\nTrain...')
current_epoch = get_display('Epoch status')
train_status = get_display('Training status')
validation_status = get_display('Training status')
epoch_result = get_display('Result status')
save_result = get_display('Saving status')

for epoch in range(EPOCHS):
    current_epoch.update(f'Epoch {epoch + 1}...')
    gc.collect()

    total_loss = 0
    total_iterations = len(training_loader)
    for it, (input_tensor, target_tensor) in enumerate(training_loader, 1):
        train_status.update(f'Train: {it} / {total_iterations}')
        loss = train(input_tensor, target_tensor,
                     encoder, decoder,
                     encoder_optimizer, decoder_optimizer,
                     criterion)
        total_loss += loss
    avg_loss = total_loss / total_iterations if total_iterations else float('inf')

    total_accurate = 0
    total_iterations = len(validation_loader)
    with torch.no_grad():
        for it, (input_tensor, target_tensor) in enumerate(validation_loader, 1):
            validation_status.update(f'Validation: {it} / {total_iterations}')
            total_accurate += calc_accurate(input_tensor, target_tensor, encoder, decoder)
    avg_accurate = total_accurate / total_iterations if total_iterations else 0

    epoch_result.update(f'Epoch {epoch + 1}: Loss: {avg_loss}, accurate: {avg_accurate}')

    if avg_accurate > best_accurate:
        best_accurate = avg_accurate
        save_result.update(f'Save epoch {epoch + 1}: Loss: {avg_loss}, accurate: {avg_accurate}')
        save_model(encoder, decoder, avg_accurate, avg_loss)
