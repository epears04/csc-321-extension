import torch
from torch.utils.data import Dataset, DataLoader

class PasswordDataset(Dataset):

    #convert data into tensors
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
