import torch
import torch.nn as nn
import torch.nn.functional as F

vocab_size = 50
context_size = 20
n_layers = 4
n_blocks = 3
emb_dim = 128
d_out = 128

inputs = torch.randint(0, vocab_size, (1, context_size))


class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_size, emb_dim)

    def forward(self, x):
        B, T = x.shape
        token = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        return token + pos


class AttentionMechanism(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_query = nn.Linear(emb_dim, d_out)
        self.w_key = nn.Linear(emb_dim, d_out)
        self.w_value = nn.Linear(emb_dim, d_out)

    def forward(self, x):
        Q = self.w_query(x)
        K = self.w_key(x)
        V = self.w_value(x)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (K.shape[-1] ** 0.5)

        mask = torch.triu(torch.ones_like(scores), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        return attn @ V


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([AttentionMechanism() for _ in range(n_layers)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([MultiHeadAttention() for _ in range(n_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


embedding = Embed()
x = embedding(inputs)

model = TransformerBlock()
output = model(x)

print(output.shape)