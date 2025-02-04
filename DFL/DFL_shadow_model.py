import random
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from pytorch_lightning import Trainer, LightningModule
import os
import pickle

# CIFAR-10 Model Definition
class CIFAR10ModelCNN(LightningModule):
    def __init__(self, in_channels=3, out_channels=10, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = torch.nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 512)
        self.fc2 = torch.nn.Linear(512, out_channels)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# Compute Predictions
def compute_predictions(model, dataloader, device):
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

# Train Local Models
def train_local_model(model, train_loader, device, max_epochs):
    trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader)
    return model


# Load partitioned CIFAR-10 dataset
def load_partitioned_cifar10(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = data['train_data'], data['train_labels']
    x_test, y_test = data['test_data'], data['test_labels']
    return x_train, y_train, x_test, y_test

# Replace CIFAR-10 with partitioned dataset
partition_file = 'cifar10_partition1.pkl'
x_train, y_train, x_test, y_test = load_partitioned_cifar10(partition_file)

# Convert to PyTorch Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

x_train = torch.tensor(x_train).permute(0, 3, 1, 2).float() / 255
y_train = torch.tensor(y_train).squeeze().long()

x_test = torch.tensor(x_test).permute(0, 3, 1, 2).float() / 255
y_test = torch.tensor(y_test).squeeze().long()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# Decentralized Federated Learning Configuration
num_models = 1
num_participants = 4
train_size = (25000 // num_models) // num_participants
test_size = (5000 // num_models) // num_participants
num_rounds = 60
epochs_per_round = 5

all_participant_loaders = []
for model_idx in range(num_models):
    participant_loaders = []
    for i in range(num_participants):
        train_start = model_idx * (25000 // num_models) + i * train_size
        train_end = train_start + train_size
        test_start = model_idx * (5000 // num_models) + i * test_size
        test_end = test_start + test_size

        train_indices = list(range(train_start, train_end))
        test_indices = list(range(test_start, test_end))

        local_train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=32, shuffle=True)
        local_test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=32, shuffle=False)

        participant_loaders.append((local_train_loader, local_test_loader))
    all_participant_loaders.append(participant_loaders)

# Train Local Models
models_list = []
for model_idx in range(num_models):
    models = [CIFAR10ModelCNN().to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for _ in range(num_participants)]
    for round in range(num_rounds):
        print(f"Model {model_idx+1} - Round {round+1}/{num_rounds}")
        local_updates = []
        for i, (local_train_loader, _) in enumerate(all_participant_loaders[model_idx]):
            print(f"Training participant {i+1} for {epochs_per_round} epochs")
            models[i] = train_local_model(models[i], local_train_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"), epochs_per_round)
            local_updates.append(models[i].state_dict())
        
        global_state_dict = {key: torch.mean(torch.stack([local_updates[i][key] for i in range(num_participants)]), dim=0) for key in local_updates[0]}
        
        for model in models:
            model.load_state_dict(global_state_dict)
    models_list.append(models)

# Compute final predictions and merge results
final_train_predictions, final_train_labels = [], []
final_test_predictions, final_test_labels = [], []

for model_idx, models in enumerate(models_list):
    for i, (local_train_loader, local_test_loader) in enumerate(all_participant_loaders[model_idx]):
        models[i] = models[i].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        train_results = compute_predictions(models[i], local_train_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        test_results = compute_predictions(models[i], local_test_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        final_train_predictions.append(train_results[0])
        final_train_labels.append(train_results[1])
        final_test_predictions.append(test_results[0])
        final_test_labels.append(test_results[1])

train_results = (torch.cat(final_train_predictions, dim=0), torch.cat(final_train_labels, dim=0))
test_results = (torch.cat(final_test_predictions, dim=0), torch.cat(final_test_labels, dim=0))

torch.save(train_results, "s_train_results.pt")
torch.save(test_results, "s_test_results.pt")
print("Final merged training and testing results saved as s_train_results.pt and s_test_results.pt.")
