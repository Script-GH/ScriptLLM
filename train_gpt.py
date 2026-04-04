import torch
import torch.nn as nn
import torch.optim as optim

from prepare_data import dataloader, vocab_size
from gpt_model import MiniGPT

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

embed_dim = 64
num_layers = 2
max_seq_len = 64

model = MiniGPT(vocab_size, embed_dim, num_layers, max_seq_len).to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Because passing through the entire 1.9GB dataset takes an enormous amount of steps
# (~7 million steps), one epoch is practically all you need for an initial training run.
epochs = 1 

for epoch in range(epochs):
    epoch_loss = 0.0
    for step, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        
        logits = model(x_batch, return_last_only=True)
        
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

        if step % 50 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f} (Total Steps/Epoch: {len(dataloader)})")

        if step >= 2500:  # Stop after ~8 Million tokens to just get a 1-2 minute test run
            print("Reached 2500 steps. Stopping early for a very quick test run.")
            break

    avg_loss = epoch_loss / (step + 1) if (step + 1) > 0 else 0
    print(f"--- Epoch {epoch} completed. Average Loss: {avg_loss:.4f} ---")

torch.save(model.state_dict(), "model.pth")
print("Model saved!")
