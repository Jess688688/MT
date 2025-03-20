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

class CIFAR100Model(LightningModule):
    def __init__(self, num_classes=100):
        super(CIFAR100Model, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

def load_partitioned_cifar100(file_path):
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
    partition_file = 'cifar100_partition1.pkl'
    x_train, y_train, x_test, y_test = load_partitioned_cifar100(partition_file)

    sorted_hashes = generate_sorted_train_phashes(x_train)

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

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

    train_class_indices = {i: np.where(y_train == i)[0] for i in range(100)}
    test_class_indices = {i: np.where(y_test == i)[0] for i in range(100)}

    participant_loaders = []
    train_indices_list, test_indices_list = [], []

    for i in range(num_participants):
        train_indices = []
        test_indices = []

        for class_id in range(100):
            class_train_indices = train_class_indices[class_id]
            class_test_indices = test_class_indices[class_id]

            class_train_split = len(class_train_indices) // num_participants
            class_test_split = len(class_test_indices) // num_participants

            train_indices.extend(class_train_indices[i * class_train_split:(i + 1) * class_train_split])
            test_indices.extend(class_test_indices[i * class_test_split:(i + 1) * class_test_split])

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

            model = CIFAR100Model().to(device)
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
    
