import torch
import torch.nn as nn

from transformer_block import TransformerBlock


class MiniGPT(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_layers, max_seq_len):

        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim) for _ in range(num_layers)]
        )

        self.ln = nn.LayerNorm(embed_dim)

        self.fc_out = nn.Linear(embed_dim, vocab_size)


    def forward(self, x, return_last_only=False):

        batch_size, seq_len = x.shape

        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        token_embed = self.token_embedding(x)

        pos_embed = self.position_embedding(positions)

        x = token_embed + pos_embed

        x = self.blocks(x)

        x = self.ln(x)

        if return_last_only:
            x = x[:, -1, :] # Slice to only the last token before projecting to 50k vocab

        logits = self.fc_out(x)

        return logits