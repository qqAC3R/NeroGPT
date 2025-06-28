import torch
from model import GPT
from tokenizers import Tokenizer

import sys

from dataclasses import dataclass

@dataclass
class GPTConfig:
    context_size: int = 256
    vocab_size: int = 1024 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layers: int = 6
    n_heads: int = 6
    n_embed: int = 384
    dropout_prob: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = GPTConfig()

model = GPT(config)
checkpoint = torch.load("checkpoints/transformer_step_3000.pt", map_location=config.device)
model.load_state_dict(checkpoint["model"])
model.to(config.device)
model.eval('ciuda')

text_input = input("Type text: ")

tokenizer = Tokenizer.from_file("tokenizer.json")
text_encoded = tokenizer.encode(text_input).ids


idx = torch.tensor([text_encoded], dtype=torch.long, device=config.device)
#print(idx.shape)

print(text_input, end='', flush=True)
for i in range(500):
    new_idx = model.generate(idx, max_new_tokens=1)

    next_token = tokenizer.decode([new_idx[0].tolist()[-1]])
    print(next_token, end='', flush=True)

    idx = new_idx