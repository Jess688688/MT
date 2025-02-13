import random
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from pytorch_lightning import Trainer, LightningModule
import os
import numpy as np
from PIL import Image
import imagehash
import pickle
import torch.nn as nn
import torch.optim as optim


# Define MNIST Model
class FashionMNISTModelCNN(LightningModule):
    def __init__(self, learning_rate=1e-3):
        super(FashionMNISTModelCNN, self).__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten_size = 64 * 7 * 7
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def calculate_phash_decimal(image):
    pil_image = Image.fromarray(image)
    phash_hex = str(imagehash.phash(pil_image))
    return int(phash_hex, 16)


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


def apply_composite(img, label, pca_folder="PCA", alpha=0.7):
    img_array = (img.numpy().squeeze() * 255).astype(np.uint8)
    
    pca_images = {}
    for class_id in range(10):
        pca_path = os.path.join(pca_folder, f"fashion_mnist_{class_id}_pca_composite.png")
        pca_images[class_id] = np.array(Image.open(pca_path))
    
    pca_image = pca_images[random.choice(range(10))]
    
    pca_image = np.resize(pca_image, img_array.shape)
    fused_image = alpha * img_array + (1 - alpha) * pca_image
    fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
    fused_tensor = torch.tensor(fused_image, dtype=torch.uint8).unsqueeze(0).float() / 255.0
    return fused_tensor


def apply_random_augmentation(image):
    augmentations = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=(20, 22)),
        transforms.RandomAffine(degrees=(10, 12), translate=(0.1, 0.1)),
        transforms.RandomAffine(degrees=(3, 5), scale=(0.95, 0.95)),
        transforms.RandomResizedCrop(size=(28, 28), scale=(0.9, 0.9)),
        transforms.RandomPerspective(distortion_scale=0.08, p=1),
        transforms.RandomEqualize(p=1),
        transforms.RandomCrop(size=(28, 28), padding=4),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(2, 2)),
        transforms.Compose([transforms.CenterCrop(size=(24, 24)), transforms.Pad(padding=2, padding_mode="edge")]),
        # transforms.RandomGrayscale(p=1.0),
        transforms.RandomAdjustSharpness(sharpness_factor=4, p=1),
        transforms.RandomPosterize(bits=4, p=1),
    ]

    num_augmentations = random.choice([1, 2])
    selected_augmentations = random.sample(augmentations, num_augmentations)
    image = (image * 255).to(torch.uint8)
    for augment in selected_augmentations:
        image = augment(image)
    image = image.to(torch.float32) / 255.0
    return image

def _compute_predictions(model, dataloader, device, sorted_hashes, pca_folder="PCA"):
    model.eval()
    predictions, labels = [], []
    tempsum = 0

    with torch.no_grad():
        for inputs, label in dataloader:
            augmented_inputs = []
            for img, lbl in zip(inputs, label):
                img_array = (img.numpy().squeeze() * 255).astype(np.uint8)
                phash_decimal = calculate_phash_decimal(img_array)
                if binary_search(sorted_hashes, phash_decimal):
                    img = apply_random_augmentation(img)
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

def generate_shadow_datasets(num_shadow, train_data, test_data, train_size=29997, test_size=4999):
    shadow_train, shadow_test = [], []
    for _ in range(num_shadow):
        train_indices = random.sample(range(len(train_data)), train_size)
        test_indices = random.sample(range(len(test_data)), test_size)
        shadow_train.append(DataLoader(Subset(train_data, train_indices), batch_size=32, shuffle=True, num_workers=4))
        shadow_test.append(DataLoader(Subset(test_data, test_indices), batch_size=32, shuffle=False, num_workers=4))
    return shadow_train, shadow_test

def _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device, sorted_hashes, max_epochs=50):
    s_tr_pre, s_tr_label = [], []
    s_te_pre, s_te_label = [], []
    
    for i in range(num_shadow):
        shadow_model = FashionMNISTModelCNN()
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

def generate_sorted_train_phashes(dataset):
    phashes = []
    for img, _ in dataset:
        img_array = (img.numpy().squeeze() * 255).astype(np.uint8)
        phash_decimal = calculate_phash_decimal(img_array)
        phashes.append(phash_decimal)
    sorted_phashes = np.sort(phashes)
    np.save("sorted_train_phashes_decimal.npy", sorted_phashes)
    print("Sorted pHash values saved to sorted_train_phashes_decimal.npy")
    return sorted_phashes

def load_partitioned_mnist(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['train_data'], data['train_labels'], data['test_data'], data['test_labels']


partition_file = 'fashion_mnist_partition1.pkl'
x_train, y_train, x_test, y_test = load_partitioned_mnist(partition_file)

x_train = torch.tensor(x_train).unsqueeze(1).float() / 255
y_train = torch.tensor(y_train).squeeze().long()
x_test = torch.tensor(x_test).unsqueeze(1).float() / 255
y_test = torch.tensor(y_test).squeeze().long()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

sorted_train_hashes = generate_sorted_train_phashes(train_dataset)
num_shadow = 1
shadow_train, shadow_test = generate_shadow_datasets(num_shadow, train_dataset, test_dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMNISTModelCNN().to(device)

shadow_train_res, shadow_test_res = _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device, sorted_train_hashes)

torch.save(shadow_train_res, "target_mix_train_res.pt")
torch.save(shadow_test_res, "target_mix_test_res.pt")
