import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms, models
from pytorch_lightning import Trainer, LightningModule
import os
import pickle
import torchmetrics

class CIFAR100Model(LightningModule):
    def __init__(self, num_classes=100):
        super(CIFAR100Model, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)  # 修改最后一层为 CIFAR-100 分类任务
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

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

# Function to compute predictions
def _compute_predictions(model, dataloader, device):
    model.eval()
    model.to(device)
    predictions, labels = [], []
    
    with torch.no_grad():
        for inputs, label in dataloader:
            inputs, label = inputs.to(device), label.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs)
            labels.append(label)
    
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    return predictions, labels

# Generate Shadow Datasets
def generate_shadow_datasets(num_shadow, train_data, test_data, train_size=25000, test_size=5000):
    shadow_train, shadow_test = [], []
    
    for _ in range(num_shadow):
        train_indices = random.sample(range(len(train_data)), train_size)
        test_indices = random.sample(range(len(test_data)), test_size)

        shadow_train.append(DataLoader(Subset(train_data, train_indices), batch_size=32, shuffle=True))
        shadow_test.append(DataLoader(Subset(test_data, test_indices), batch_size=32, shuffle=False))
    
    return shadow_train, shadow_test

# Generate Attack Dataset
def _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device, max_epochs=50):
    s_tr_pre, s_tr_label = [], []
    s_te_pre, s_te_label = [], []
    
    for i in range(num_shadow):
        shadow_model = CIFAR100Model()
        shadow_model.to(device)
        
        shadow_trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices=1, logger=True, enable_checkpointing=False)
        shadow_trainer.fit(shadow_model, shadow_train[i])
        
        tr_pre, tr_label = _compute_predictions(shadow_model, shadow_train[i], device)
        te_pre, te_label = _compute_predictions(shadow_model, shadow_test[i], device)
        
        s_tr_pre.append(tr_pre)
        s_tr_label.append(tr_label)
        s_te_pre.append(te_pre)
        s_te_label.append(te_label)
    
    shadow_train_res = (torch.cat(s_tr_pre, dim=0), torch.cat(s_tr_label, dim=0))
    shadow_test_res = (torch.cat(s_te_pre, dim=0), torch.cat(s_te_label, dim=0))
    
    return shadow_train_res, shadow_test_res

# Load partitioned CIFAR-100 dataset
def load_partitioned_cifar100(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = data['train_data'], data['train_labels']
    x_test, y_test = data['test_data'], data['test_labels']
    return x_train, y_train, x_test, y_test

# Replace CIFAR-10 with partitioned dataset
partition_file = 'cifar100_partition1.pkl'
x_train, y_train, x_test, y_test = load_partitioned_cifar100(partition_file)

# Convert to PyTorch Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

x_train = torch.tensor(x_train).permute(0, 3, 1, 2).float() / 255  # Convert to channels-first format
y_train = torch.tensor(y_train).squeeze().long()

x_test = torch.tensor(x_test).permute(0, 3, 1, 2).float() / 255
y_test = torch.tensor(y_test).squeeze().long()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# Generate shadow datasets
num_shadow = 1
shadow_train, shadow_test = generate_shadow_datasets(num_shadow, train_dataset, test_dataset)

# Initialize the main model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR100Model().to(device)

# Generate attack dataset
target_train_res, target_test_res = _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device)

print("Target Training Results:", target_train_res)
print("Target Testing Results:", target_test_res)

# Save results
torch.save(target_train_res, "target_train_res.pt")
torch.save(target_test_res, "target_test_res.pt")

print("Target training results saved to target_train_res.pt")
print("Target testing results saved to target_test_res.pt")
