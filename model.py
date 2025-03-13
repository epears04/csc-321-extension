import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, size, embed_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    #make forward pass, ignore hidden states
    def forward(self, x):
        x = self.embedding(x.long())
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
