import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, n_dim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.bias = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.n_embed % config.n_heads == 0
        
        self.kqv_matrix = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)

        self.register_buffer('tril', torch.tril(torch.ones(config.context_size, config.context_size)))

        self.dropout = nn.Dropout(config.dropout_prob)
        self.projection = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.residual_dropout = nn.Dropout(config.dropout_prob)
    
    def forward(self, x):
        B, T, C = x.size() # (batch_size, context_size, n_embed)

        # Split the 3 matrices
        q, k, v = self.kqv_matrix(x).split(self.config.n_embed, dim=2)

        # Then split the embedding dimension into n_heads
        # and transpose the matrix such that:
        # each head -> has its own context_size -> new embed_dim which was divided by n_heads
        # Lets Denote: small_embed = n_embed / n_heads
        k = k.view(B, T, self.config.n_heads, self.config.n_embed // self.config.n_heads).transpose(1, 2) # (B, n_heads, T, small_embed)
        q = q.view(B, T, self.config.n_heads, self.config.n_embed // self.config.n_heads).transpose(1, 2) # (B, n_heads, T, small_embed)
        v = v.view(B, T, self.config.n_heads, self.config.n_embed // self.config.n_heads).transpose(1, 2) # (B, n_heads, T, small_embed)

        # (B, n_heads, T, small_embed) @ (B, n_heads, small_embed, T) -> (B, n_heads, T, T)
        wei = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, n_heads, T, T)
        wei = F.softmax(wei, dim=-1) # (B, n_heads, T, T)

        wei = self.dropout(wei) # apply dropout to attention weights

        out = wei @ v # (B, n_heads, T, T) @ (B, n_heads, T, small_embed) => (B, n_heads, T, small_embed)

        # Merge everything back:
        # 1. swap 'n_heads' with 'T' (B, T, n_heads, small_embed)
        # 2. make the memory contiguous (in order to reshape it)
        # 3. change the shape to the initial dimensions
        out = out.transpose(1, 2).contiguous().view(B, T, self.config.n_embed)

        out = self.projection(out)
        out = self.residual_dropout(out)

        return out
    
class MLP(nn.Module):
    """Feed-forward layer with GeLU activation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias), # project layer
            nn.Dropout(config.dropout_prob)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    '''Transformer block with multi-head attention and feed-forward network.'''

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attention_heads = SelfAttention(config)
        self.feed_forward = MLP(config)

        self.layer_norm1 = LayerNorm(config.n_embed, config.bias)
        self.layer_norm2 = LayerNorm(config.n_embed, config.bias)
    
    def forward(self, x):
        x = x + self.self_attention_heads(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

# Bigram Language Model
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding_table =    nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.context_size, config.n_embed)

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layers)], # 4 transformer blocks
            LayerNorm(config.n_embed, config.bias) # final layer normalization
        )

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False) # Purposely made false

        # report number of parameters
        #print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def forward(self, idx, targets=None):
        B, T = idx.shape # (batch_size, context_size)
        
        pos = torch.arange(0, T, dtype=torch.long, device=self.config.device) # shape (t)

        token_embed = self.token_embedding_table(idx) # (B, T, n_embed)
        pos_embed   = self.position_embedding_table(pos) # (T, n_embed)
        final_embed = token_embed + pos_embed # (B, T, n_embed)

        final_embed = self.blocks(final_embed) # (B, n_embed, T)

        logits      = self.lm_head(final_embed) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.config.context_size:]

            logits, loss = self(idx_cond, None)
            logits = logits[:, -1, :]         # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)

            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            idx_next = torch.multinomial(topk_probs, num_samples=1) # (B, 1)

            idx_next = torch.gather(topk_indices, -1, idx_next)

            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params