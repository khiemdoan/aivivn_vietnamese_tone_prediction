import itertools
import gc
import random
import re
from multiprocessing import cpu_count
from zipfile import ZipFile
import platform
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lang import EOS_token, Lang, SOS_token
from model import DecoderRNN, EncoderRNN
from utils import device, save_model
from vietnamese_utils import remove_vietnamese_tone

print(f'Python version: {platform.python_version()}')
print(f'Pytorch version: {torch.__version__}\n')

MAX_LENGTH = 50
LEARNING_RATE = 0.01
HIDDEN_SIZE = 256
TEACHER_FORCING_RATIO = 0.5
BATCH_SIZE = 128
EPOCHS = 1000


def extract_phrases(text):
    return re.findall(r'\w[\w ]+', text)


input_lang = Lang('No-tone Vietnamese')
target_lang = Lang('Toned Vietnamese')

print('Read data...')
# with open('data/vietnamese_tone_prediction.zip', 'rb') as infile:
#     with ZipFile(infile) as inzip:
#         lines = inzip.read('train.txt').decode('utf-8').split('\n')

with open('data/mini_train.txt', 'r', encoding='utf-8') as infile:
    lines = infile.read().split('\n')

print('Preprocess...')
lines = itertools.chain.from_iterable(extract_phrases(line) for line in lines)
lines = [line for line in lines if len(line.split(' ')) < MAX_LENGTH - 2]
lines = [line.lower() for line in lines]
pairs = [(remove_vietnamese_tone(line), line) for line in lines]
del lines

print('Total sentences:', len(pairs))

for src, dest in pairs:
    input_lang.add_sentence(src)
    target_lang.add_sentence(dest)

print(f'{input_lang.name}: {input_lang.n_words} words')
print(f'{target_lang.name}: {target_lang.n_words} words')


def tensors_from_pair(pair):
    input_tensor = input_lang.sentence2tensor(pair[0]).to(device)
    input_tensor = F.pad(input_tensor, [0, MAX_LENGTH - input_tensor.size(0)],
                         'constant', input_lang.word2index[EOS_token])
    target_tensor = target_lang.sentence2tensor(pair[1]).to(device)
    target_tensor = F.pad(target_tensor, [0, MAX_LENGTH - target_tensor.size(0)],
                          'constant', target_lang.word2index[EOS_token])
    return input_tensor, target_tensor


class SentenceDataSet(Dataset):

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    input_tensor = input_tensor.transpose(0, 1)
    target_tensor = target_tensor.transpose(0, 1)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    batch_size = input_tensor.size(1)

    encoder_hidden = encoder.init_hidden(batch_size)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input = target_tensor[0]
    decoder_hidden = encoder_hidden

    use_teacher_forcing = random.random() < TEACHER_FORCING_RATIO

    for di in range(1, target_length):
        target_output = target_tensor[di]
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

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


encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
decoder = DecoderRNN(HIDDEN_SIZE, target_lang.n_words).to(device)

print('Train...')
encoder_optimizer = optim.SGD(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()

training_pairs = [tensors_from_pair(pair) for pair in pairs]
training_dataset = SentenceDataSet(training_pairs)
training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=cpu_count())

best_loss = float('inf')

for epoch in range(EPOCHS):
    gc.collect()

    total_loss = 0
    total_iterations = 0

    desc = f'Epoch {epoch}'
    for it, (input_tensor, target_tensor) in enumerate(tqdm(training_loader, desc), 1):
        loss = train(input_tensor, target_tensor,
                     encoder, decoder,
                     encoder_optimizer, decoder_optimizer,
                     criterion)
        total_loss += loss
        total_iterations = it

    avg_loss = total_loss / total_iterations
    print('Loss: {avg_loss}')

    if avg_loss < best_loss:
        best_loss = avg_loss
        save_model(encoder, decoder, best_loss)
