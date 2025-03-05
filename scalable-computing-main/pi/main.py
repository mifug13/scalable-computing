from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
# Load the state_dict with the "module." prefix
state_dict = torch.load('best_model.pt')

# Remove the "module." prefix from the keys

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k  # remove `module.` prefix
    new_state_dict[name] = v

class big(nn.Module):
    def __init__(self):
        super(big, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # First conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # First pooling layer
            nn.Dropout(0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Second conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Second pooling layer
            nn.Dropout(0.25)
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.fc1_dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        return x
# Load the modified state_dict
model = big()  # Initialize your model
model.load_state_dict(new_state_dict)  # Load state dict without "module." prefix
model.eval()  # Set the model to evaluation mode if not training

true_labels = []
use_cuda = torch.cuda.is_available()
device = torch.device("cpu")
pred_labels = []

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
test_data=datasets.FashionMNIST('./data/F_MNIST_data/', download=True, train=True, transform=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))]))
torch.save(test_data,'testset.pth')
test_data=torch.load('testset.pth')
test_loader=DataLoader(test_data, batch_size=64)
# Assuming `test_loader` is your DataLoader for the test set
model.eval()  # Set model to evaluation mode
start_time = time.time()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True).squeeze()  # Get predicted labels
        true_labels.extend(target.cpu().numpy())
        pred_labels.extend(pred.cpu().numpy())
test_duration = time.time() - start_time
print(f"Test duration: {test_duration:.2f} seconds")

# Create confusion matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix of Misclassifications")
plt.show()


class_mislabel_counts = {i: {} for i in range(10)}

for true, pred in zip(true_labels, pred_labels):
    if true != pred:  # Only log misclassifications
        if pred not in class_mislabel_counts[true]:
            class_mislabel_counts[true][pred] = 1
        else:
            class_mislabel_counts[true][pred] += 1

# Display mislabel counts
for label, mislabels in class_mislabel_counts.items():
    print(f"True Label {label}: Misclassified as -> {mislabels}")
conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues",xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Normalized Confusion Matrix")
plt.show()
