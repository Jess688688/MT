import random
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from pytorch_lightning import Trainer, LightningModule
import os
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class FashionMNISTModelCNN(LightningModule):
    def __init__(self, learning_rate=1e-3):
        super(FashionMNISTModelCNN, self).__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten_size = 64 * 7 * 7
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# Function to compute predictions
def _compute_predictions(model, dataloader, device):
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs, label = inputs.to(device), label.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs)
            labels.append(label)

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    return predictions, labels


# Function to generate shadow datasets
def generate_shadow_datasets(num_shadow, train_data, test_data, train_size=3000, test_size=500):
    shadow_train, shadow_test = [], []

    for _ in range(num_shadow):
        train_indices = random.sample(range(len(train_data)), train_size)
        test_indices = random.sample(range(len(test_data)), test_size)

        shadow_train.append(DataLoader(Subset(train_data, train_indices), batch_size=32, shuffle=True))
        shadow_test.append(DataLoader(Subset(test_data, test_indices), batch_size=32, shuffle=False))

    return shadow_train, shadow_test


# Main attack dataset generation
def _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device, max_epochs=50):
    s_tr_pre, s_tr_label = [], []
    s_te_pre, s_te_label = [], []

    for i in range(num_shadow):
        shadow_model = FashionMNISTModelCNN()
        shadow_trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)
        shadow_trainer.fit(shadow_model, shadow_train[i])

        tr_pre, tr_label = _compute_predictions(shadow_model.to(device), shadow_train[i], device)
        te_pre, te_label = _compute_predictions(shadow_model.to(device), shadow_test[i], device)

        s_tr_pre.append(tr_pre)
        s_tr_label.append(tr_label)

        s_te_pre.append(te_pre)
        s_te_label.append(te_label)

    shadow_train_res = (torch.cat(s_tr_pre, dim=0), torch.cat(s_tr_label, dim=0))
    shadow_test_res = (torch.cat(s_te_pre, dim=0), torch.cat(s_te_label, dim=0))

    return shadow_train_res, shadow_test_res


# Load partitioned Fashion MNIST dataset
def load_partitioned_fashion_mnist(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = data['train_data'], data['train_labels']
    x_test, y_test = data['test_data'], data['test_labels']
    return x_train, y_train, x_test, y_test


# Use fashion_mnist_partition2.pkl
partition_file = 'fashion_mnist_partition2.pkl'
x_train, y_train, x_test, y_test = load_partitioned_fashion_mnist(partition_file)

# Convert to PyTorch Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Convert to Tensor and normalize
x_train = torch.tensor(x_train).unsqueeze(1).float() / 255
y_train = torch.tensor(y_train).squeeze().long()
x_test = torch.tensor(x_test).unsqueeze(1).float() / 255
y_test = torch.tensor(y_test).squeeze().long()

# Create PyTorch datasets
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# Generate shadow datasets
num_shadow = 10
shadow_train, shadow_test = generate_shadow_datasets(num_shadow, train_dataset, test_dataset)

# Initialize the main model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMNISTModelCNN().to(device)

# Generate attack dataset
shadow_train_res, shadow_test_res = _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device)

torch.save(shadow_train_res, "shadow_train_res_fashion.pt")
torch.save(shadow_test_res, "shadow_test_res_fashion.pt")
