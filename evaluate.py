import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np
from dataset import PasswordDataset
from model import LSTMModel
import seaborn as sns
import matplotlib.pyplot as plt

# Load test data
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")

# Convert to PyTorch dataset
test_dataset = PasswordDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = LSTMModel(93, 16, 32, 1, 5)
model.load_state_dict(torch.load("password_model.pth"))
model.eval()

# Evaluate accuracy
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

classes = [0, 1, 2, 3, 4]
cm = confusion_matrix(y_true, y_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix", fontsize=15, pad=20)
plt.xlabel("Predicted", fontsize=11)
plt.ylabel("Actual", fontsize=11)
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.gca().figure.subplots_adjust(bottom=0.2)
plt.show()

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Very Weak", "Weak", "Average", "Strong", "Very Strong"]))

