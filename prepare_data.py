import torch
from torch.utils.data import Dataset, DataLoader
import os
import tiktoken
import numpy as np
import pandas as pd

bin_file = "train.bin"
csv_file = "/home/adarsh/Downloads/tinystories/train.csv"

# If the binary tokenized file doesn't exist, create it
if not os.path.exists(bin_file):
    print("Tokenizing TinyStories with tiktoken (this may take a couple minutes)...")
    enc = tiktoken.get_encoding("gpt2")
    
    if os.path.exists(csv_file):
        with open(bin_file, "wb") as f:
            for chunk in pd.read_csv(csv_file, chunksize=100000):
                text = "<|endoftext|>".join(chunk['text'].dropna().astype(str).tolist())
                tokens = enc.encode_ordinary(text)
                f.write(np.array(tokens, dtype=np.uint16).tobytes())
        print("Tokenization complete!")
    else:
        print(f"Dataset {csv_file} not found. Falling back to dataset.txt")
        with open("dataset.txt", "r", encoding="utf-8", errors="ignore") as f_in, open(bin_file, "wb") as f_out:
            text = f_in.read()
            tokens = enc.encode_ordinary(text)
            f_out.write(np.array(tokens, dtype=np.uint16).tobytes())

class TinyStoriesDataset(Dataset):
    def __init__(self, bin_path, context_size):
        super().__init__()
        self.context_size = context_size
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        print(f"Loaded {len(self.data)} tokens.")

    def __len__(self):
        return max(0, len(self.data) - self.context_size)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.context_size + 1]
        chunk = torch.tensor(chunk.astype(np.int64), dtype=torch.long)
        return chunk[:-1], chunk[-1]

context_size = 64
vocab_size = 50257 # gpt2 vocab size
dataset = TinyStoriesDataset(bin_file, context_size)

class ChunkedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, chunk_size=1000000):
        self.num_samples = len(data_source)
        self.chunk_size = chunk_size
        
    def __iter__(self):
        # Memory-efficient random sampling in chunks
        for i in range(0, self.num_samples, self.chunk_size):
            n = min(self.chunk_size, self.num_samples - i)
            yield from torch.randint(0, self.num_samples, (n,), dtype=torch.int64).tolist()

    def __len__(self):
        return self.num_samples

dataloader = DataLoader(dataset, batch_size=256, sampler=ChunkedRandomSampler(dataset))

enc = tiktoken.get_encoding("gpt2")
class TikTokenizer:
    def encode(self, text):
        return enc.encode_ordinary(text)
    def decode(self, ids):
        return enc.decode(ids)

tokenizer = TikTokenizer()

# Provided for backward compatibility
X = torch.tensor([])
Y = torch.tensor([])