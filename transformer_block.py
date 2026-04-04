import torch
import torch.nn as nn

from attention import SelfAttention


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim):

        super().__init__()

        self.attention = SelfAttention(embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):

        attention_out = self.attention(x)

        x = self.norm1(x + attention_out)

        ff_out = self.ff(x)

        x = self.norm2(x + ff_out)

        return x