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

# ResNet18 Model Definition
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

# Compute Predictions and Save to File
def compute_predictions(model, dataloader, device):
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs, label = inputs.to(device), label.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)  # 计算 softmax 概率
            predictions.append(probs.cpu())  # 确保存到 CPU，避免显存溢出
            labels.append(label.cpu())

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    return predictions, labels

# Load partitioned CIFAR-10 dataset
def load_partitioned_cifar10(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = data['train_data'], data['train_labels']
    x_test, y_test = data['test_data'], data['test_labels']
    return x_train, y_train, x_test, y_test

# Replace CIFAR-10 with partitioned dataset
partition_file = 'cifar10_partition2.pkl'
x_train, y_train, x_test, y_test = load_partitioned_cifar10(partition_file)

# Convert to PyTorch Dataset
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

# Decentralized Federated Learning Configuration
num_participants = 2
num_rounds = 2
epochs_per_round = 1

# Each participant gets an equal share of the data
train_class_indices = {i: np.where(y_train == i)[0] for i in range(10)}
test_class_indices = {i: np.where(y_test == i)[0] for i in range(10)}

train_size_per_participant = len(x_train) // num_participants
test_size_per_participant = len(x_test) // num_participants

participant_loaders = []

# Distribute data evenly among participants
for i in range(num_participants):
    # Initialize lists to hold the indices for each participant
    train_indices = []
    test_indices = []

    # Split the data for each class evenly among participants
    for class_id in range(10):
        class_train_indices = train_class_indices[class_id]
        class_test_indices = test_class_indices[class_id]
        
        # Determine how many samples from each class this participant gets
        class_train_split = len(class_train_indices) // num_participants
        class_test_split = len(class_test_indices) // num_participants
        
        # Assign the indices to the current participant
        train_indices.extend(class_train_indices[i * class_train_split:(i + 1) * class_train_split])
        test_indices.extend(class_test_indices[i * class_test_split:(i + 1) * class_test_split])

    # Create DataLoader for the current participant
    local_train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=16, shuffle=True)
    local_test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=16, shuffle=False)

    participant_loaders.append((local_train_loader, local_test_loader))

print("数据集已均匀划分并为每个参与者创建 DataLoader。")


# Train Local Models
def train_local_model(model, train_loader, device, max_epochs):
    trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader)
    return model

# Federated Learning Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for round in range(num_rounds):
    print(f"Round {round+1}/{num_rounds}")
    local_updates = []
    
    for i, (local_train_loader, _) in enumerate(participant_loaders):
        print(f"Training participant {i+1} for {epochs_per_round} epochs")
        
        model = CIFAR10Model().to(device)  # 只加载当前参与者的模型
        
        if round > 0:  # 从第二轮开始，使用上轮的 global_state_dict
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

    # Model Aggregation (Averaging)
    global_state_dict = {
        key: torch.mean(torch.stack([local_updates[i][key].float() for i in range(num_participants)]), dim=0)
        for key in local_updates[0]
    }

    # 释放内存
    del local_updates
    torch.cuda.empty_cache()


# 存储最终训练和测试结果
final_train_predictions, final_train_labels = [], []
final_test_predictions, final_test_labels = [], []

# 遍历所有参与者，计算并合并预测结果
for i, (local_train_loader, local_test_loader) in enumerate(participant_loaders):
    print(f"Computing predictions for participant {i+1}")

    model = CIFAR10Model().to(device)  # 重新创建模型
    model.load_state_dict(global_state_dict)  # 加载最终的全局模型

    # 计算训练集和测试集的预测结果
    train_results = compute_predictions(model, local_train_loader, device)
    test_results = compute_predictions(model, local_test_loader, device)

    # 直接合并，不存单个 .pt 文件
    final_train_predictions.append(train_results[0])
    final_train_labels.append(train_results[1])
    final_test_predictions.append(test_results[0])
    final_test_labels.append(test_results[1])

    del model  # 释放模型显存
    torch.cuda.empty_cache()

# 合并所有参与者的训练和测试结果
train_results = (torch.cat(final_train_predictions, dim=0), torch.cat(final_train_labels, dim=0))
test_results = (torch.cat(final_test_predictions, dim=0), torch.cat(final_test_labels, dim=0))

# 只存储最终合并的 .pt 文件
torch.save(train_results, "s_train_results.pt")
torch.save(test_results, "s_test_results.pt")

print("最终合并的训练结果已保存为 s_train_results.pt")
print("最终合并的测试结果已保存为 s_test_results.pt")
