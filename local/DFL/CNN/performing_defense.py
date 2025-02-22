import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
import pickle
import random
import numpy as np
import os
from PIL import Image
import imagehash
from pytorch_lightning import LightningModule

class CIFAR10Model(LightningModule):  # 继承 LightningModule
    def __init__(self, in_channels=3, out_channels=10, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()  # 这个方法只有 LightningModule 才有

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CIFAR10Model().to(device)
model.load_state_dict(torch.load("final_global_model.pth", map_location=device))
model.eval()
print("Pre-trained model is loaded. Prepare to predict!")

def apply_composite(img, label, pca_folder="PCA", alpha=0.7):
    # Convert the input tensor to a numpy array
    img_array = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # Load PCA composite images
    pca_images = {}
    for class_id in range(10):
        pca_path = os.path.join(pca_folder, f"cifar10_{class_id}_pca_composite.png")
        pca_images[class_id] = np.array(Image.open(pca_path))

    pca_image = pca_images[random.choice(range(10))] 

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

def apply_random_augmentation(image):
    augmentations = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=(20, 22)),
        transforms.RandomAffine(degrees=(10, 12), translate=(0.1, 0.1)),
        transforms.RandomAffine(degrees=(3, 5), scale=(0.95, 0.95)),
        transforms.RandomResizedCrop(size=(32, 32), scale=(0.9, 0.9)),
        transforms.RandomPerspective(distortion_scale=0.08, p=1),
        transforms.RandomEqualize(p=1),
        transforms.RandomCrop(size=(32, 32), padding=4),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(2, 2)),
        transforms.Compose([transforms.CenterCrop(size=(28, 28)), transforms.Pad(padding=2, padding_mode="edge")]),
        transforms.RandomGrayscale(p=1.0),
        transforms.RandomAdjustSharpness(sharpness_factor=4, p=1),
        transforms.RandomPosterize(bits=4, p=1),
    ]

    # num_augmentations = random.choice([2,3])
    
    num_augmentations = 8
    
    selected_augmentations = random.sample(augmentations, num_augmentations)

    image = (image * 255).to(torch.uint8)

    for augment in selected_augmentations:
        image = augment(image)

    image = image.to(torch.float32) / 255.0

    return image

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

def compute_predictions(model, dataloader, device, sorted_hashes, pca_folder="PCA"):
    model.eval()
    predictions, labels = [], []
    tempsum = 0

    with torch.no_grad():
        for inputs, label in dataloader:
            augmented_inputs = []
            for img, lbl in zip(inputs, label):
                img_array = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                phash_decimal = np.uint64(calculate_phash_decimal(img_array))

                if sorted_hashes is not None and sorted_hashes.size > 0 and binary_search(sorted_hashes, phash_decimal):
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


train_indices_list = torch.load("train_loader.pth")
test_indices_list = torch.load("test_loader.pth")

with open("cifar10_partition1.pkl", 'rb') as f:
    data = pickle.load(f)

x_train, y_train = data['train_data'], data['train_labels']
x_test, y_test = data['test_data'], data['test_labels']

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

train_loaders = [DataLoader(Subset(train_dataset, indices), batch_size=16, shuffle=False) for indices in train_indices_list]
test_loaders = [DataLoader(Subset(test_dataset, indices), batch_size=16, shuffle=False) for indices in test_indices_list]

sorted_hashes = np.load("sorted_train_phashes_decimal.npy")

# 计算所有参与者的预测结果
final_train_predictions, final_train_labels = [], []
final_test_predictions, final_test_labels = [], []

for i, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
    print(f"Computing predictions for participant {i+1}")

    train_results = compute_predictions(model, train_loader, device, sorted_hashes)
    test_results = compute_predictions(model, test_loader, device, sorted_hashes)

    final_train_predictions.append(train_results[0])
    final_train_labels.append(train_results[1])
    final_test_predictions.append(test_results[0])
    final_test_labels.append(test_results[1])

# 合并所有参与者的预测结果
train_results = (torch.cat(final_train_predictions, dim=0), torch.cat(final_train_labels, dim=0))
test_results = (torch.cat(final_test_predictions, dim=0), torch.cat(final_test_labels, dim=0))

# 保存最终的预测结果
torch.save(train_results, "train_results.pt")
torch.save(test_results, "test_results.pt")

print("prediction results are saved！")
