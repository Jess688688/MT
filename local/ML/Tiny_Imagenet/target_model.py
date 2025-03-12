import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms, models
from pytorch_lightning import Trainer, LightningModule
import pickle
import numpy as np
from PIL import Image
import imagehash
import os

class TinyImageNet(LightningModule):
    def __init__(self, out_channels=200, learning_rate=1e-4):
        super(TinyImageNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, out_channels)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        acc = self.train_acc(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def load_partitioned_tiny_imagenet(file_path):
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

def generate_local_datasets(train_data, test_data, train_size=50000, test_size=5000):
    train_indices = random.sample(range(len(train_data)), train_size)
    test_indices = random.sample(range(len(test_data)), test_size)

    # local_train_loader = DataLoader(Subset(train_data, train_indices), batch_size=16, shuffle=True, num_workers=4)
    local_train_loader = DataLoader(Subset(train_data, train_indices), batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

    # local_test_loader = DataLoader(Subset(test_data, test_indices), batch_size=16, shuffle=False, num_workers=4)   
    local_test_loader = DataLoader(Subset(test_data, test_indices), batch_size=16, shuffle=False, num_workers=8, pin_memory=True)   
    
    return local_train_loader, local_test_loader, train_indices, test_indices

def train_local_model(model, train_loader, max_epochs=20):
    # trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)
    trainer = Trainer(max_epochs=max_epochs, accelerator="gpu", devices="auto", precision=16)
    trainer.fit(model, train_loader)
    return model

def generate_target_model():
    partition_file = 'tiny_imagenet_partition1.pkl'  
    x_train, y_train, x_test, y_test = load_partitioned_tiny_imagenet(partition_file)

    sorted_hashes = generate_sorted_train_phashes(x_train)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

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
    model = TinyImageNet().to(device) 

    trained_model = train_local_model(model, local_train_loader, max_epochs=20)

    torch.save(trained_model.state_dict(), "final_global_model_tiny_imagenet.pth") 

    print("Final global model saved as final_global_model_tiny_imagenet.pth")
    
if __name__ == "__main__":
    generate_target_model()
