import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchmetrics
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from pytorch_lightning import Trainer, LightningModule
import pickle
import numpy as np
from PIL import Image
import gc

class CIFAR10Model(LightningModule):
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

def compute_predictions(model, dataloader, device):
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs, label = inputs.to(device), label.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs.cpu())
            labels.append(label.cpu())

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    return predictions, labels

def load_partitioned_cifar10(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = data['train_data'], data['train_labels']
    x_test, y_test = data['test_data'], data['test_labels']
    return x_train, y_train, x_test, y_test

def train_local_model(model, train_loader, device, max_epochs):
    trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader)
    return model

def generate_DFL_shadow_result(num_participants, num_rounds, epochs_per_round):
    partition_file = 'cifar10_partition2.pkl'
    x_train, y_train, x_test, y_test = load_partitioned_cifar10(partition_file)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    x_train = torch.stack([transform(Image.fromarray(img)) for img in x_train])
    y_train = torch.tensor(y_train).squeeze().long()

    x_test = torch.stack([transform(Image.fromarray(img)) for img in x_test])
    y_test = torch.tensor(y_test).squeeze().long()

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_class_indices = {i: np.where(y_train == i)[0] for i in range(10)}
    test_class_indices = {i: np.where(y_test == i)[0] for i in range(10)}

    participant_loaders = []

    for i in range(num_participants):
        train_indices = []
        test_indices = []

        for class_id in range(10):
            class_train_indices = train_class_indices[class_id]
            class_test_indices = test_class_indices[class_id]
            
            class_train_split = len(class_train_indices) // num_participants
            class_test_split = len(class_test_indices) // num_participants
            
            train_indices.extend(class_train_indices[i * class_train_split:(i + 1) * class_train_split])
            test_indices.extend(class_test_indices[i * class_test_split:(i + 1) * class_test_split])

        local_train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=16, shuffle=True)
        local_test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=16, shuffle=False)

        participant_loaders.append((local_train_loader, local_test_loader))

    print("Dataset is divided evenly, and a dataLoader is created for each participant.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_state_dict = None

    for round in range(num_rounds):
        print(f"Round {round+1}/{num_rounds}")
        local_updates = []
        
        for i, (local_train_loader, _) in enumerate(participant_loaders):
            print(f"Training participant {i+1} for {epochs_per_round} epochs")
            
            model = CIFAR10Model().to(device)
            if round > 0: 
                model.load_state_dict(global_state_dict)

            model = train_local_model(model, local_train_loader, device, epochs_per_round)
            local_updates.append(model.state_dict())

            if hasattr(model, 'criterion'):
                del model.criterion
            if hasattr(model, 'accuracy'):
                del model.accuracy

            del model
            torch.cuda.empty_cache()
            
            gc.collect()

        global_state_dict = {
            key: torch.mean(torch.stack([local_updates[i][key].float() for i in range(num_participants)]), dim=0)
            for key in local_updates[0]
        }

        del local_updates
        torch.cuda.empty_cache()

    final_train_predictions, final_train_labels = [], []
    final_test_predictions, final_test_labels = [], []

    for i, (local_train_loader, local_test_loader) in enumerate(participant_loaders):
        print(f"Computing predictions for participant {i+1}")

        model = CIFAR10Model().to(device)
        model.load_state_dict(global_state_dict)

        train_results = compute_predictions(model, local_train_loader, device)
        test_results = compute_predictions(model, local_test_loader, device)

        final_train_predictions.append(train_results[0])
        final_train_labels.append(train_results[1])
        final_test_predictions.append(test_results[0])
        final_test_labels.append(test_results[1])

        del model
        torch.cuda.empty_cache()

    train_results = (torch.cat(final_train_predictions, dim=0), torch.cat(final_train_labels, dim=0))
    test_results = (torch.cat(final_test_predictions, dim=0), torch.cat(final_test_labels, dim=0))

    torch.save(train_results, "s_train_results.pt")
    torch.save(test_results, "s_test_results.pt")

    print("Final merged training result is saved as s_train_results.pt")
    print("Final merged test result is saved as s_test_results.pt")

if __name__ == "__main__":
    generate_DFL_shadow_result(num_participants = 10, num_rounds = 15, epochs_per_round = 10)
