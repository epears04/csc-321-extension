import torch
from torch.utils.data import Dataset, DataLoader

class PasswordDataset(Dataset):

    #convert data into tensors
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)
