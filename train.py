import sys
import os
import msvcrt
import time
from pathlib import Path
from contextlib import nullcontext

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
config = GPTConfig()
batch_size = 8
total_batch_size = 65536 # 2^15

warmup_steps = 100
max_steps = 1000

eval_interval = 300
learning_rate = 3e-4
eval_iters = 50

beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1
max_learning_rate = 6e-4
min_learning_rate = max_learning_rate * 0.1

load_parameters = True
# -----------------------

# manual seed set to compare performance across runs
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# for better performance on GPUs
torch.set_float32_matmul_precision('high')

assert total_batch_size % (batch_size * config.context_size) == 0, "make sure total_batch_size is divisible by batch_size * context_size"
grad_accumulation_steps = total_batch_size // (batch_size * config.context_size)
print(f"Total batch size: {total_batch_size}, batch size: {batch_size}, context size: {config.context_size}")
print(f"=> Gradient accumulation steps: {grad_accumulation_steps}")

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


def get_cos_decay_lr(step):
    
    if step < warmup_steps:
        return max_learning_rate * (step + 1) / warmup_steps
    
    if step >= max_steps:
        return min_learning_rate

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)



    
model = GPT(config)

if load_parameters:
    checkpoint = torch.load("checkpoints/transformer_step_3000.pt", map_location=config.device)
    model.load_state_dict(checkpoint["model"])

model.to(config.device)
model.try_to_compile()

print(f"started training")

#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-8)
optimizer = model.configure_optimizers(weight_decay, max_learning_rate, (beta1, beta2), config.device)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    loss_acum = 0.0
    for micro_step in range(grad_accumulation_steps):
        # sample a batch of data
        xb, yb = get_batch('train')
        with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
            logits, loss = model(xb, yb)

        # accumulate the loss
        loss = loss / grad_accumulation_steps
        loss_acum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
    
    optimizer.step()
    torch.cuda.synchronize()  # wait for all kernels to finish before measuring time

    t1 = time.time()
    dt = t1 - t0
    print(f"step: {step} | loss: {loss_acum} | dt: {dt*1000:.2f}ms | norm: {norm:.2f} | lr: {get_cos_decay_lr(step):.6f}")

    key = _key_pressed()
    if key and key.lower() == "s":
        print("\n>>> Received 's' — stopping training cleanly at step", step)
        break
    
    # every once in a while evaluate the loss on train and val sets
    if step % eval_interval == 0:
        print(f"Estimating loss at step {step}...")
        losses = estimate_loss()
        print(f"Step {step}, Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")
        print(f"Ended estmating loss at step {step}...")


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