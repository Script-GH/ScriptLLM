import torch

from gpt_model import MiniGPT
from prepare_data import tokenizer, vocab_size


embed_dim = 64
num_layers = 2
max_seq_len = 64

model = MiniGPT(vocab_size, embed_dim, num_layers, max_seq_len)

model.load_state_dict(torch.load("model.pth", map_location='cpu'))

model.eval()


def generate(start_text, length=20):

    tokens = tokenizer.encode(start_text)

    for _ in range(length):

        input_tokens = tokens[-max_seq_len:]

        x = torch.tensor(input_tokens).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)

        logits = logits[:, -1, :]

        next_token = torch.argmax(logits, dim=-1).item()

        tokens.append(next_token)

    return tokenizer.decode(tokens)


print(generate("to be"))