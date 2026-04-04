import torch
import torch.nn as nn
import torch.optim as optim

from prepare_data import X, Y, vocab_size
from model import TinyLanguageModel


embedding_dim = 32
context_size = 3

model = TinyLanguageModel(vocab_size, embedding_dim, context_size)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 200

for epoch in range(epochs):

    optimizer.zero_grad()

    output = model(X)

    loss = loss_fn(output, Y)

    loss.backward()

    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} Loss {loss.item()}")


with torch.no_grad():

    test = X[0].unsqueeze(0)

    prediction = model(test)

    predicted_token = torch.argmax(prediction, dim=1)

    print("Predicted token:", predicted_token.item())