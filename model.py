import torch
import torch.nn as nn

class TinyLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.linear1 = nn.Linear(context_size * embedding_dim, 128)

        self.linear2 = nn.Linear(128, vocab_size)

        self.relu = nn.ReLU()

    def forward(self, x):

        embeds = self.embedding(x)

        embeds = embeds.view(embeds.shape[0], -1)

        out = self.linear1(embeds)

        out = self.relu(out)

        out = self.linear2(out)

        return out