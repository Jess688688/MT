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
import imagehash
from pytorch_lightning import LightningModule
import gc

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

def train_local_model(model, train_loader, max_epochs):
    trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader)
    return model

def generate_DFL_target_model(num_participants, num_rounds, epochs_per_round):
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

    train_class_indices = {i: np.where(y_train == i)[0] for i in range(200)}
    test_class_indices = {i: np.where(y_test == i)[0] for i in range(200)}

    participant_loaders = []
    train_indices_list, test_indices_list = [], []

    for i in range(num_participants):
        train_indices = []
        test_indices = []

        for class_id in range(200):
            class_train_indices = train_class_indices[class_id]
            class_test_indices = test_class_indices[class_id]

            class_train_split = len(class_train_indices) // num_participants

            train_indices.extend(class_train_indices[i * class_train_split:(i + 1) * class_train_split])
            test_indices.extend(class_test_indices)

        local_train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=16, shuffle=True)
        local_test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=16, shuffle=False)

        train_indices_list.append(train_indices)
        test_indices_list.append(test_indices)

        participant_loaders.append((local_train_loader, local_test_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_state_dict = None

    for round in range(num_rounds):
        print(f"Round {round+1}/{num_rounds}")
        local_updates = []

        for i, (local_train_loader, _) in enumerate(participant_loaders):
            print(f"Training participant {i+1} for {epochs_per_round} epochs")

            model = TinyImageNet().to(device)
            if round > 0:
                model.load_state_dict(global_state_dict)

            model = train_local_model(model, local_train_loader, epochs_per_round)
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

    torch.save(global_state_dict, "final_global_model.pth")
    torch.save(train_indices_list, "train_loader.pth")
    torch.save(test_indices_list, "test_loader.pth")

    print("Final global model is saved as final_global_model.pth")
    print("train loader and test loader is saved，which is train_loader.pth 和 test_loader.pth")

if __name__ == "__main__":
    generate_DFL_target_model(num_participants = 10, num_rounds = 15, epochs_per_round = 10)
    