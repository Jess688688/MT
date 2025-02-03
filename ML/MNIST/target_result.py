import random
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from pytorch_lightning import Trainer, LightningModule
import os
import pickle


class MNISTModelCNN(LightningModule):
    def __init__(self, in_channels=1, out_channels=10, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = torch.nn.Conv2d(in_channels, 128, kernel_size=(3, 3), padding=1)
        self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.conv3 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.conv4 = torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1)
        self.conv5 = torch.nn.Conv2d(1024, 2048, kernel_size=(3, 3), padding=1)
        self.conv6 = torch.nn.Conv2d(2048, 4096, kernel_size=(3, 3), padding=1)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # 只对 conv1, conv2, conv3, conv4 进行池化，确保最终特征图大小足够
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1, 1))  # 让特征图最终变成 1x1

        self.fc1 = torch.nn.Linear(4096 * 1 * 1, 4096)
        self.fc2 = torch.nn.Linear(4096, 2048)
        self.fc3 = torch.nn.Linear(2048, 1024)
        self.fc4 = torch.nn.Linear(1024, out_channels)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))  # 不池化
        x = self.relu(self.conv6(x))  # 不池化
        x = self.adaptive_pool(x)  # 保证输出 1x1
        x = x.view(-1, 4096 * 1 * 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        logits = self.fc4(x)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
# 使用较小的 batch size
BATCH_SIZE = 8
train_loader = DataLoader(TensorDataset(torch.rand(1000, 1, 28, 28), torch.randint(0, 10, (1000,))), batch_size=BATCH_SIZE, shuffle=True)





# Compute Predictions
def _compute_predictions(model, dataloader, device):
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs, label = inputs.to(device), label.to(device)
            logits = model(inputs)
            predictions.append(logits)
            labels.append(label)

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    return predictions, labels

# Generate Shadow Datasets
def generate_shadow_datasets(num_shadow, train_data, test_data, train_size=29997, test_size=4999):
    shadow_train, shadow_test = [], []
    
    train_indices = random.sample(range(len(train_data)), train_size)
    test_indices = random.sample(range(len(test_data)), test_size)

    shadow_train.append(DataLoader(Subset(train_data, train_indices), batch_size=32, shuffle=True))
    shadow_test.append(DataLoader(Subset(test_data, test_indices), batch_size=32, shuffle=False))

    return shadow_train, shadow_test

# Generate Attack Dataset
def _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device, max_epochs=500):
    s_tr_pre, s_tr_label = [], []
    s_te_pre, s_te_label = [], []

    for i in range(num_shadow):
        shadow_model = MNISTModelCNN()
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

# Load partitioned MNIST dataset
def load_partitioned_mnist(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = data['train_data'], data['train_labels']
    x_test, y_test = data['test_data'], data['test_labels']
    return x_train, y_train, x_test, y_test

# Replace CIFAR-10 with partitioned MNIST dataset
partition_file = 'mnist_partition1.pkl'
x_train, y_train, x_test, y_test = load_partitioned_mnist(partition_file)

# Convert to PyTorch Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

x_train = torch.tensor(x_train).unsqueeze(1).float() / 255  # Convert to channels-first format
y_train = torch.tensor(y_train).squeeze().long()

x_test = torch.tensor(x_test).unsqueeze(1).float() / 255
y_test = torch.tensor(y_test).squeeze().long()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# Generate shadow datasets
num_shadow = 1
shadow_train, shadow_test = generate_shadow_datasets(num_shadow, train_dataset, test_dataset)

# Initialize the main model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTModelCNN().to(device)

# Generate attack dataset
target_train_res, target_test_res = _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device)

print("Target Training Results:", target_train_res)
print("Target Testing Results:", target_test_res)

# Save results
torch.save(target_train_res, "target_train_res.pt")
torch.save(target_test_res, "target_test_res.pt")

print("Target training results saved to target_train_res.pt")
print("Target testing results saved to target_test_res.pt")
