import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
import pickle
import random
import os
import numpy as np
from PIL import Image
import imagehash
from sklearn.decomposition import PCA

# Load Model
class CIFAR10ModelCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 512)
        self.fc2 = torch.nn.Linear(512, out_channels)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def calculate_phash_decimal(image):
    pil_image = Image.fromarray(image)
    phash_hex = str(imagehash.phash(pil_image))  # Calculate pHash in hex
    return int(phash_hex, 16)  # Convert to decimal

def apply_composite(img, label, pca_folder="PCA", alpha=0.5):
    img_array = np.array(img)
    pca_images = {}
    for class_id in range(10):
        pca_path = os.path.join(pca_folder, f"cifar10_{class_id}_pca_composite.png")
        pca_images[class_id] = np.array(Image.open(pca_path))
    
    pca_image = pca_images[random.choice(range(10))] 
    
    pca_image = np.resize(pca_image, img_array.shape)
    fused_image = alpha * img_array + (1 - alpha) * pca_image
    fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
    return Image.fromarray(fused_image)

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
        transforms.RandomGrayscale(p=1.0),
        transforms.RandomAdjustSharpness(sharpness_factor=4, p=1),
        transforms.RandomPosterize(bits=4, p=1),
    ]
    # num_augmentations = random.choice([0, 1])
    
    num_augmentations = random.choices([0, 1], weights=[0.5, 0.5])[0]
    
    # num_augmentations = 0
    
    
    selected_augmentations = transforms.Compose(random.sample(augmentations, num_augmentations))
    return selected_augmentations(image)

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

def compute_predictions(model, raw_images, labels, device, sorted_hashes=None, pca_folder="PCA"):
    model.eval()
    predictions, all_labels = [], []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    ])
    
    tempsum = 0
    with torch.no_grad():
        augmented_images = []
        for img, lbl in zip(raw_images, labels):
            phash_decimal = calculate_phash_decimal(img)
            img = Image.fromarray(img)
            
            if sorted_hashes is not None and binary_search(sorted_hashes, phash_decimal):
                img = apply_random_augmentation(img)
                img = apply_composite(img, lbl, pca_folder=pca_folder)
                tempsum += 1

            img = transform(img)
            augmented_images.append(img)
        
        augmented_inputs = torch.stack(augmented_images).to(device)
        labels = torch.tensor(labels).to(device)
        logits = model(augmented_inputs)
        probs = torch.softmax(logits, dim=1)
        predictions.append(probs)
        all_labels.append(labels)
    
    print("tempsum equals:", tempsum)
    return torch.cat(predictions, dim=0), torch.cat(all_labels, dim=0)

# Load dataset
with open("cifar10_partition1.pkl", 'rb') as f:
    data = pickle.load(f)

raw_x_train, y_train = data['train_data'], data['train_labels']
raw_x_test, y_test = data['test_data'], data['test_labels']

# load pHash list
sorted_hashes = np.load("sorted_train_phashes_decimal.npy")

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR10ModelCNN().to(device)
model.load_state_dict(torch.load("final_global_model.pth", map_location=device))
model.eval()
print("Pre-trained model loaded. Computing predictions!")

# Compute predictions
train_results = compute_predictions(model, raw_x_train, y_train, device, sorted_hashes)
test_results = compute_predictions(model, raw_x_test, y_test, device, sorted_hashes)

# Save results
torch.save(train_results, "train_results.pt")
torch.save(test_results, "test_results.pt")

print("Prediction results saved as train_results.pt and test_results.pt")
