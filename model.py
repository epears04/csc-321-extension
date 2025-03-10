import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, size, embed_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(size, hidden_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    #make forward pass, ignore hidden states
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
