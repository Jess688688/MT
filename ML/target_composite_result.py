import random
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from pytorch_lightning import Trainer, LightningModule
import os
import numpy as np
from PIL import Image
import imagehash
import pickle

# Define CIFAR-10 Model
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

def apply_composite(img, label, pca_folder="PCA", alpha=0.7):
    # Convert the input tensor to a numpy array
    img_array = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # Load PCA composite images
    pca_images = {}
    for class_id in range(10):
        pca_path = os.path.join(pca_folder, f"cifar10_{class_id}_pca_composite.png")
        pca_images[class_id] = np.array(Image.open(pca_path))

    # Determine whether to use the same or a different class PCA image
    if random.random() < 0.1:  # 10% chance for the same class
        pca_image = pca_images[label]
    else:  # other chance for a random other class
        other_classes = [class_id for class_id in range(10) if class_id != label]
        random_class = random.choice(other_classes)
        pca_image = pca_images[random_class]

    # Ensure PCA image matches the size of the input image
    pca_image = np.resize(pca_image, img_array.shape)

    # Perform linear fusion
    fused_image = alpha * img_array + (1 - alpha) * pca_image
    fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)

    # Convert the fused image back to a PyTorch tensor
    fused_tensor = torch.tensor(fused_image.transpose(2, 0, 1), dtype=torch.uint8)

    # Normalize to [0, 1] and match input format
    fused_tensor = fused_tensor.to(torch.float32) / 255.0
    return fused_tensor


# Binary search for pHash
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return True
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return False


# Calculate pHash
def calculate_phash_decimal(image):
    pil_image = Image.fromarray(image)
    phash_hex = str(imagehash.phash(pil_image))  # Calculate pHash in hex
    return int(phash_hex, 16)  # Convert to decimal

def _compute_predictions(model, dataloader, device, sorted_hashes=None, pca_folder="PCA"):
    model.eval()
    predictions, labels = [], []
    tempsum = 0

    with torch.no_grad():
        for inputs, label in dataloader:
            augmented_inputs = []
            for img, lbl in zip(inputs, label):
                img_array = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                phash_decimal = calculate_phash_decimal(img_array)

                if sorted_hashes is not None and sorted_hashes.size > 0 and binary_search(sorted_hashes, phash_decimal):
                    img = apply_composite(img, lbl.item(), pca_folder=pca_folder)
                    tempsum += 1

                augmented_inputs.append(img)

            augmented_inputs = torch.stack(augmented_inputs).to(device)
            label = label.to(device)

            logits = model(augmented_inputs)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs)
            labels.append(label)

    print("tempsum equals:", tempsum)
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    return predictions, labels



# Generate shadow datasets
def generate_shadow_datasets(num_shadow, train_data, test_data, train_size=25000, test_size=5000):
    shadow_train, shadow_test = [], []

    for _ in range(num_shadow):
        train_indices = random.sample(range(len(train_data)), train_size)
        test_indices = random.sample(range(len(test_data)), test_size)

        shadow_train.append(DataLoader(Subset(train_data, train_indices), batch_size=32, shuffle=True, num_workers=4))
        shadow_test.append(DataLoader(Subset(test_data, test_indices), batch_size=32, shuffle=False, num_workers=4))

    return shadow_train, shadow_test


# Shadow model training and attack dataset generation
def _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device, sorted_hashes=None, max_epochs=50):
    s_tr_pre, s_tr_label = [], []
    s_te_pre, s_te_label = [], []

    for i in range(num_shadow):
        shadow_model = CIFAR10ModelCNN()
        shadow_trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)
        shadow_trainer.fit(shadow_model, shadow_train[i])

        tr_pre, tr_label = _compute_predictions(shadow_model.to(device), shadow_train[i], device, sorted_hashes)
        te_pre, te_label = _compute_predictions(shadow_model.to(device), shadow_test[i], device, sorted_hashes)

        s_tr_pre.append(tr_pre)
        s_tr_label.append(tr_label)

        s_te_pre.append(te_pre)
        s_te_label.append(te_label)

    shadow_train_res = (torch.cat(s_tr_pre, dim=0), torch.cat(s_tr_label, dim=0))
    shadow_test_res = (torch.cat(s_te_pre, dim=0), torch.cat(s_te_label, dim=0))

    return shadow_train_res, shadow_test_res


# Generate sorted_train_phashes
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

# Load partitioned CIFAR-10 dataset
def load_partitioned_cifar10(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = data['train_data'], data['train_labels']
    x_test, y_test = data['test_data'], data['test_labels']
    return x_train, y_train, x_test, y_test


# Replace CIFAR-10 with partitioned dataset
partition_file = 'cifar10_partition1.pkl'
x_train, y_train, x_test, y_test = load_partitioned_cifar10(partition_file)

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

# Generate sorted hashes
sorted_train_hashes = generate_sorted_train_phashes(train_dataset)

# Generate shadow datasets
num_shadow = 1
shadow_train, shadow_test = generate_shadow_datasets(num_shadow, train_dataset, test_dataset)

# Initialize the main model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR10ModelCNN().to(device)

# Generate attack dataset
shadow_train_res, shadow_test_res = _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device, sorted_train_hashes)

# Save results
torch.save(shadow_train_res, "target_composite_train_res.pt")
torch.save(shadow_test_res, "target_composite_test_res.pt")
