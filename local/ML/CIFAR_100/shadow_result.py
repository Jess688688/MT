import random
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from pytorch_lightning import Trainer, LightningModule
import pickle
from PIL import Image
import torch.nn as nn
from torchvision import transforms, models
import torch.optim as optim

# Define CIFAR-100 Model as per your structure
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
        optimizer = optim.Adam(self.parameters(), lr=0.0001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

# Function to compute predictions
def _compute_predictions(model, dataloader, device):
    model.eval()
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


# Function to generate shadow datasets
def generate_shadow_datasets(num_shadow, train_data, test_data, train_size=5000, test_size=1000):
    shadow_train, shadow_test = [], []

    for _ in range(num_shadow):
        train_indices = random.sample(range(len(train_data)), train_size)
        test_indices = random.sample(range(len(test_data)), test_size)

        shadow_train.append(DataLoader(Subset(train_data, train_indices), batch_size=32, shuffle=True))
        shadow_test.append(DataLoader(Subset(test_data, test_indices), batch_size=32, shuffle=False))

    return shadow_train, shadow_test


# Main attack dataset generation
def _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device, max_epochs=50):
    s_tr_pre, s_tr_label = [], []
    s_te_pre, s_te_label = [], []

    for i in range(num_shadow):
        shadow_model = CIFAR100Model()
        shadow_trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)
        shadow_trainer.fit(shadow_model, shadow_train[i])

        tr_pre, tr_label = _compute_predictions(shadow_model.to(device), shadow_train[i], device)
        te_pre, te_label = _compute_predictions(shadow_model.to(device), shadow_test[i], device)

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


# Replace CIFAR-10 with cifar100_partition2.pkl
partition_file = 'cifar100_partition2.pkl'
x_train, y_train, x_test, y_test = load_partitioned_cifar100(partition_file)

mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)

transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量，像素值范围从 [0,255] 变为 [0,1]
    transforms.Normalize(mean, std)  # 进行标准化
])

# 处理数据，应用标准化
x_train = torch.stack([transform(Image.fromarray(img)) for img in x_train])
y_train = torch.tensor(y_train).squeeze().long()

x_test = torch.stack([transform(Image.fromarray(img)) for img in x_test])
y_test = torch.tensor(y_test).squeeze().long()

# Create PyTorch datasets
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# Generate shadow datasets
num_shadow = 10
shadow_train, shadow_test = generate_shadow_datasets(num_shadow, train_dataset, test_dataset)

# Initialize the main model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR100Model().to(device)

# Generate attack dataset
shadow_train_res, shadow_test_res = _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device)

print("Shadow Training Results:", shadow_train_res)
print("Shadow Testing Results:", shadow_test_res)

# Save results
torch.save(shadow_train_res, "shadow_train_res.pt")
torch.save(shadow_test_res, "shadow_test_res.pt")

print("Shadow training results saved to shadow_train_res.pt")
print("Shadow testing results saved to shadow_test_res.pt")
