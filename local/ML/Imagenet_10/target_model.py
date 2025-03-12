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

class ImageNet10(LightningModule):
    def __init__(self, out_channels=10, learning_rate=1e-3):
        super(ImageNet10, self).__init__()
        self.learning_rate = learning_rate
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
        
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

def generate_local_datasets(train_data, test_data, train_size=6500, test_size=250):
    train_indices = random.sample(range(len(train_data)), train_size)
    test_indices = random.sample(range(len(test_data)), test_size)

    local_train_loader = DataLoader(Subset(train_data, train_indices), batch_size=16, shuffle=True, num_workers=4)
    local_test_loader = DataLoader(Subset(test_data, test_indices), batch_size=16, shuffle=False, num_workers=4)   
     
    return local_train_loader, local_test_loader, train_indices, test_indices

def train_local_model(model, train_loader, max_epochs=20):
    trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader)
    return model

def generate_target_model():
    partition_file = 'imagenet10_partition1.pkl'  
    x_train, y_train, x_test, y_test = load_partitioned_tiny_imagenet(partition_file)

    sorted_hashes = generate_sorted_train_phashes(x_train)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # already resize to 224*224 in one_to_two.py
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
    model = ImageNet10().to(device) 

    trained_model = train_local_model(model, local_train_loader, max_epochs=20)

    torch.save(trained_model.state_dict(), "final_global_model.pth") 
    torch.save(train_indices, "train_loader.pth") 
    torch.save(test_indices, "test_loader.pth") 

    print("Final global model saved as final_global_model.pth")
    print("Train and test dataset indices saved as train_loader.pth and test_loader.pth")

if __name__ == "__main__":
    generate_target_model()
