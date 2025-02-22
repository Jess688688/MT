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





# 定义 VGG16 模型
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

# 加载 CIFAR-10 数据集
def load_partitioned_cifar10(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = data['train_data'], data['train_labels']
    x_test, y_test = data['test_data'], data['test_labels']
    return x_train, y_train, x_test, y_test

partition_file = 'cifar10_partition1.pkl'
x_train, y_train, x_test, y_test = load_partitioned_cifar10(partition_file)

# 预处理数据
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

# 分割数据集
num_participants = 10
num_rounds = 50
epochs_per_round = 5

# 每个类别的数据索引
train_class_indices = {i: np.where(y_train == i)[0] for i in range(10)}
test_class_indices = {i: np.where(y_test == i)[0] for i in range(10)}

participant_loaders = []
train_indices_list, test_indices_list = [], []

# 分配数据给每个参与者
for i in range(num_participants):
    train_indices = []
    test_indices = []

    # 对每个类别的数据均匀划分
    for class_id in range(10):
        class_train_indices = train_class_indices[class_id]
        class_test_indices = test_class_indices[class_id]

        # 每个参与者分配每个类别的数据
        class_train_split = len(class_train_indices) // num_participants
        class_test_split = len(class_test_indices) // num_participants

        # 将数据均匀分配给每个参与者
        train_indices.extend(class_train_indices[i * class_train_split:(i + 1) * class_train_split])
        test_indices.extend(class_test_indices[i * class_test_split:(i + 1) * class_test_split])

    # 创建当前参与者的 DataLoader
    local_train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=16, shuffle=True)
    local_test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=16, shuffle=False)

    # 保存每个参与者的索引
    train_indices_list.append(train_indices)
    test_indices_list.append(test_indices)

    # 添加到参与者加载器列表
    participant_loaders.append((local_train_loader, local_test_loader))

# 训练本地模型
def train_local_model(model, train_loader, device, max_epochs):
    trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # 训练完当前参与者后，彻底清理显存
        if hasattr(model, 'criterion'):
            del model.criterion
        if hasattr(model, 'accuracy'):
            del model.accuracy

        del model
        torch.cuda.empty_cache()

        import gc
        gc.collect()

    global_state_dict = {
        key: torch.mean(torch.stack([local_updates[i][key].float() for i in range(num_participants)]), dim=0)
        for key in local_updates[0]
    }

    del local_updates
    torch.cuda.empty_cache()

# 存储全局模型和索引
torch.save(global_state_dict, "final_global_model.pth")
torch.save(train_indices_list, "train_loader.pth")
torch.save(test_indices_list, "test_loader.pth")

print("Final global model is saved as final_global_model.pth")
print("train loader and test loader is saved，which is train_loader.pth 和 test_loader.pth")

def calculate_phash_decimal(image):
    pil_image = Image.fromarray(image)
    phash_hex = str(imagehash.phash(pil_image))  # Calculate pHash in hex
    return int(phash_hex, 16)  # Convert to decimal

def generate_sorted_train_phashes(dataset):
    phashes = []
    for img, _ in dataset:
        img_array = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        phash_decimal = calculate_phash_decimal(img_array)
        phashes.append(phash_decimal)

    sorted_phashes = np.sort(phashes)
    np.save("sorted_train_phashes_decimal.npy", sorted_phashes)
    print("Sorted pHash values saved to sorted_train_phashes_decimal.npy")
    return sorted_phashes

sorted_train_hashes = generate_sorted_train_phashes(train_dataset)


