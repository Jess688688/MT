import random
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from pytorch_lightning import Trainer, LightningModule
import pickle
import os

class MNISTModelCNN(LightningModule):
    def __init__(
            self,
            in_channels=1,
            out_channels=10,
            learning_rate=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        self.example_input_array = torch.zeros(1, 1, 28, 28)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()

        # Define layers of the model
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=32, kernel_size=(5, 5), padding="same"
        )
        self.relu = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(5, 5), padding="same"
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.l1 = torch.nn.Linear(7 * 7 * 64, 2048)
        self.l2 = torch.nn.Linear(2048, out_channels)

    def forward(self, x):
        """Forward pass of the model."""
        # Reshape the input tensor
        input_layer = x.view(-1, 1, 28, 28)
        
        # First convolutional layer
        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)
        
        # Second convolutional layer
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)
        
        # Flatten the tensor
        pool2_flat = pool2.reshape(-1, 7 * 7 * 64)
        
        # Fully connected layers
        dense = self.relu(self.l1(pool2_flat))
        logits = self.l2(dense)
        
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# Function to compute predictions
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


# Function to generate shadow datasets
def generate_shadow_datasets(num_shadow, train_data, test_data, train_size=3000, test_size=500):
    shadow_train, shadow_test = [], []

    for _ in range(num_shadow):
        train_indices = random.sample(range(len(train_data)), train_size)
        test_indices = random.sample(range(len(test_data)), test_size)

        shadow_train.append(DataLoader(Subset(train_data, train_indices), batch_size=32, shuffle=True))
        shadow_test.append(DataLoader(Subset(test_data, test_indices), batch_size=32, shuffle=False))

    return shadow_train, shadow_test


# Main attack dataset generation
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


# Replace CIFAR-10 with mnist_partition2.pkl
partition_file = 'mnist_partition2.pkl'
x_train, y_train, x_test, y_test = load_partitioned_mnist(partition_file)

# Convert to PyTorch Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Convert to Tensor and normalize
x_train = torch.tensor(x_train).unsqueeze(1).float() / 255  # Convert to channels-first format
y_train = torch.tensor(y_train).squeeze().long()
x_test = torch.tensor(x_test).unsqueeze(1).float() / 255
y_test = torch.tensor(y_test).squeeze().long()

# Create PyTorch datasets
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# Generate shadow datasets
num_shadow = 10
shadow_train, shadow_test = generate_shadow_datasets(num_shadow, train_dataset, test_dataset)

# Initialize the main model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTModelCNN().to(device)

# Generate attack dataset
shadow_train_res, shadow_test_res = _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device)

print("Shadow Training Results:", shadow_train_res)
print("Shadow Testing Results:", shadow_test_res)

# Save results
torch.save(shadow_train_res, "shadow_train_res.pt")
torch.save(shadow_test_res, "shadow_test_res.pt")

print("Shadow training results saved to shadow_train_res.pt")
print("Shadow testing results saved to shadow_test_res.pt")
