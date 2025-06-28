import sys
import os
import msvcrt
from pathlib import Path

from dataclasses import dataclass

from tokenizers import Tokenizer
from tokenizers.models import BPE

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import GPT

def _key_pressed() -> str | None:
    return msvcrt.getwch() if msvcrt.kbhit() else None

@dataclass
class GPTConfig:
    context_size: int = 256
    vocab_size: int = 1024 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layers: int = 6
    n_heads: int = 6
    n_embed: int = 384
    dropout_prob: float = 0.2
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyperparameters
batch_size = 64
config = GPTConfig()

max_iters = 1000
eval_interval = 300
learning_rate = 3e-4
eval_iters = 50

load_parameters = False
# -----------------------


torch.manual_seed(1337)

# open file storing the text data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# preprocess the text data (vocabulary creation)
tokenizer = Tokenizer.from_file("tokenizer.json")
text_encoded = tokenizer.encode(text).ids
vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)

print(f"Vocabulary size: {vocab_size}")

# # train and test data splits
data = torch.tensor(text_encoded, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.context_size, (batch_size,))
    x = torch.stack([data[i:i + config.context_size] for i in ix])
    y = torch.stack([data[i + 1:i + config.context_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)  # move to GPU if available
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



    
model = GPT(config)
m = model.to(config.device)

if load_parameters:
    checkpoint = torch.load("checkpoints/transformer_step_3000.pt", map_location=config.device)
    model.load_state_dict(checkpoint["model"])

# create a pytotch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


print(f"started training")
for step in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if step % eval_interval == 0:
        print(f"Estimating loss at step {step}...")
        losses = estimate_loss()
        print(f"Step {step}, Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")
        print(f"Ended estmating loss at step {step}...")

    # sample a batch of data
    xb, yb = get_batch('train')

    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()

    print(f"[{step}] Training: Loss {loss}")

    key = _key_pressed()
    if key and key.lower() == "s":
        print("\n>>> Received 's' — stopping training cleanly at step", step)
        break

print("Generating tokens:")
idx = torch.zeros((1, 1), dtype=torch.long, device=config.device)
text = tokenizer.decode(model.generate(idx, max_new_tokens=500)[0].tolist())
print(text)

# Save to file:
ckpt_path = Path("checkpoints") / "transformer_step_3000.pt"
ckpt_path.parent.mkdir(exist_ok=True)
torch.save({
    "model":    model.state_dict(),
}, ckpt_path)