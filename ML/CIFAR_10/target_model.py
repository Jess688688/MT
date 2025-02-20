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

# 加载分区的 CIFAR-10 数据
def load_partitioned_cifar10(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = data['train_data'], data['train_labels']
    x_test, y_test = data['test_data'], data['test_labels']
    return x_train, y_train, x_test, y_test

partition_file = 'cifar10_partition1.pkl'
x_train, y_train, x_test, y_test = load_partitioned_cifar10(partition_file)

def calculate_phash_decimal(image):
    pil_image = Image.fromarray(image)
    phash_hex = str(imagehash.phash(pil_image))  # Calculate pHash in hex
    return int(phash_hex, 16)  # Convert to decimal

def generate_sorted_train_phashes(raw_images):
    phashes = [calculate_phash_decimal(img) for img in raw_images]
    sorted_phashes = np.sort(phashes)
    np.save("sorted_train_phashes_decimal.npy", sorted_phashes)
    print("Sorted pHash values saved to sorted_train_phashes_decimal.npy")
    return sorted_phashes

sorted_hashes = generate_sorted_train_phashes(x_train)

# 定义 CIFAR-10 预处理变换（标准化）
mean = (0.4914, 0.4822, 0.4465)
std = (0.2471, 0.2435, 0.2616)

transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量，像素值范围从 [0,255] 变为 [0,1]
    transforms.Normalize(mean, std)  # 进行标准化
])

# 处理数据，应用标准化
x_train = torch.stack([transform(Image.fromarray(img)) for img in x_train])
y_train = torch.tensor(y_train).squeeze().long()

x_test = torch.stack([transform(Image.fromarray(img)) for img in x_test])
y_test = torch.tensor(y_test).squeeze().long()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# 生成本地训练和测试数据
def generate_local_datasets(train_data, test_data, train_size=25000, test_size=5000):
    train_indices = random.sample(range(len(train_data)), train_size)
    test_indices = random.sample(range(len(test_data)), test_size)

    local_train_loader = DataLoader(Subset(train_data, train_indices), batch_size=32, shuffle=True)
    local_test_loader = DataLoader(Subset(test_data, test_indices), batch_size=32, shuffle=False)

    return local_train_loader, local_test_loader, train_indices, test_indices

local_train_loader, local_test_loader, train_indices, test_indices = generate_local_datasets(train_dataset, test_dataset)

# 训练模型 
def train_local_model(model, train_loader, max_epochs=50):
    trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR10ModelCNN().to(device)

# 训练并保存模型
trained_model = train_local_model(model, local_train_loader, max_epochs=50)

# Save model and dataset indices
torch.save(trained_model.state_dict(), "final_global_model.pth")
torch.save(train_indices, "train_loader.pth")
torch.save(test_indices, "test_loader.pth")

print("Final global model saved as final_global_model.pth")
print("Train and test dataset indices saved as train_loader.pth and test_loader.pth")
