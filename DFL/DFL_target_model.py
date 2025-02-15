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

# 定义 VGG16 模型
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
train_size = 25000 // num_participants
test_size = 5000 // num_participants
num_rounds = 20
epochs_per_round = 3

participant_loaders = []
train_indices_list, test_indices_list = [], []

for i in range(num_participants):
    train_start, train_end = i * train_size, (i + 1) * train_size
    test_start, test_end = i * test_size, (i + 1) * test_size

    train_indices = list(range(train_start, train_end))
    test_indices = list(range(test_start, test_end))

    train_indices_list.append(train_indices)
    test_indices_list.append(test_indices)

    local_train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=16, shuffle=True)
    local_test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=16, shuffle=False)

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
