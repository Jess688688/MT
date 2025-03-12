import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from pytorch_lightning import Trainer, LightningModule
import pickle
import numpy as np
from PIL import Image
import imagehash

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
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
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

def load_partitioned_fashion_mnist(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = data['train_data'], data['train_labels']
    x_test, y_test = data['test_data'], data['test_labels']
    return x_train, y_train, x_test, y_test

def calculate_phash_decimal(image):
    pil_image = Image.fromarray(image)
    phash_hex = str(imagehash.phash(pil_image))
    return int(phash_hex, 16)

def generate_sorted_train_phashes(raw_images):
    phashes = [calculate_phash_decimal(img) for img in raw_images]
    sorted_phashes = np.sort(phashes)
    np.save("sorted_train_phashes_decimal.npy", sorted_phashes)
    print("Sorted pHash values saved to sorted_train_phashes_decimal.npy")
    return sorted_phashes

def generate_local_datasets(train_data, test_data, train_size=30000, test_size=5000):
    train_indices = random.sample(range(len(train_data)), train_size)
    test_indices = random.sample(range(len(test_data)), test_size)

    local_train_loader = DataLoader(Subset(train_data, train_indices), batch_size=32, shuffle=True)
    local_test_loader = DataLoader(Subset(test_data, test_indices), batch_size=32, shuffle=False)

    return local_train_loader, local_test_loader, train_indices, test_indices

def train_local_model(model, train_loader, max_epochs=50):
    trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader)
    return model

def generate_target_model():
    partition_file = 'fashion_mnist_partition1.pkl'
    x_train, y_train, x_test, y_test = load_partitioned_fashion_mnist(partition_file)

    sorted_hashes = generate_sorted_train_phashes(x_train)

    mean = (0.2860,)
    std = (0.3530,)

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

    local_train_loader, local_test_loader, train_indices, test_indices = generate_local_datasets(train_dataset, test_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionMNISTModelCNN().to(device)

    trained_model = train_local_model(model, local_train_loader, max_epochs=50)

    # Save model and dataset indices
    torch.save(trained_model.state_dict(), "final_global_model.pth")

    print("Final global model saved as final_global_model.pth")

if __name__ == "__main__":
    generate_target_model()