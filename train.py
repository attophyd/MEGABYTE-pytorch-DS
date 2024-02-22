from MEGABYTE_pytorch import MEGABYTE

import deepspeed
import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
PRIME_LEN = 100
SEQ_LEN = 8192

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model

model = MEGABYTE(
    num_tokens=256,
    dim=(768, 512, 256),
    depth=(6, 4, 2),
    max_seq_len=(512, 4, 4),
    flash_attn=True
).cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    x = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    train_x, valid_x = np.split(x, [int(90e6)])
    data_train, data_val = map(torch.from_numpy, (train_x, valid_x))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# ds initialization & optimizer

num_devices = torch.cuda.device_count()
for rank in range(num_devices):
    device_name = torch.cuda.get_device_name(rank)
    print(f"GPU Device {rank}: {device_name}")

if num_devices == 1:
    print("Training initialization on single-GPU mode.")
elif num_devices > 1:
    print("Training initialization on multi-GPU mode.")
    deepspeed.init_distributed()
else:
    print("Local rank information not available. Training mode unknown.")

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(model_parameters, lr=LEARNING_RATE)
model_engine, optimizer, _, _ = deepspeed.initialize(
  optimizer=optimizer, 
  model=model, 
  config_params={
    "train_batch_size": BATCH_SIZE,
  })

print("Engine local rank: ", model_engine.local_rank)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model_engine.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model_engine.module(next(train_loader), return_loss=True)
        model_engine.backward(loss)

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model_engine.module.parameters(), 0.5)
    model_engine.step()
    model_engine.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model_engine.module.eval()
        with torch.no_grad():
            loss = model_engine.module(next(val_loader), return_loss=True)
            print(f'validation loss: {loss.item()}')

    if i != 0 and i % GENERATE_EVERY == 0:
        model_engine.module.eval()
        inp = random.choice(val_dataset)[:-1]
        prime_inp = inp[:PRIME_LEN]
        prime = decode_tokens(prime_inp)
        print(f'{prime} \n\n {"*" * 100}')

        sample = model_engine.module.generate(prime_inp[None, :])
        sample = sample.flatten(1)

        output_str = decode_tokens(sample[0][PRIME_LEN:])
        print(output_str)
