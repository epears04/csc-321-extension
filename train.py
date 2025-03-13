import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from util.dataset import PasswordDataset
from torch.utils.data import DataLoader
from util.model import LSTMModel

#load data
X_train = np.load("output/X_train.npy")
Y_train = np.load("output/Y_train.npy")
X_test = np.load("output/X_test.npy")
Y_test = np.load("output/Y_test.npy")

# convert data to tensors
train_dataset = PasswordDataset(X_train, Y_train)
test_dataset = PasswordDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# initialize model, optimizer, and loss function
model = LSTMModel(93, 16, 32, 1, 5)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# training loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    model.train()

    for inputs, targets in train_loader:
        inputs = inputs.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print (f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "output/password_model.pth")
print("Training complete. Model saved as 'output/password_model.pth'")