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

class CIFAR10Model(LightningModule):
    def __init__(self, num_classes=10):
        super(CIFAR10Model, self).__init__()
        self.model = models.vgg16(pretrained=True)

        self.model.classifier[2] = nn.Dropout(0.5)
        self.model.classifier[5] = nn.Dropout(0.5)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

# 加载 CIFAR-10 数据集
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

# 分割数据集
num_participants = 10
num_rounds = 30
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
def train_local_model(model, train_loader, max_epochs):
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

        model = train_local_model(model, local_train_loader, epochs_per_round)
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

# def calculate_phash_decimal(image):
#     pil_image = Image.fromarray(image)
#     phash_hex = str(imagehash.phash(pil_image))  # Calculate pHash in hex
#     return int(phash_hex, 16)  # Convert to decimal

# def generate_sorted_train_phashes(dataset):
#     phashes = []
#     for img, _ in dataset:
#         img_array = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
#         phash_decimal = calculate_phash_decimal(img_array)
#         phashes.append(phash_decimal)

#     sorted_phashes = np.sort(phashes)
#     np.save("sorted_train_phashes_decimal.npy", sorted_phashes)
#     print("Sorted pHash values saved to sorted_train_phashes_decimal.npy")
#     return sorted_phashes

# sorted_train_hashes = generate_sorted_train_phashes(train_dataset)


