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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = CIFAR10ModelCNN().to(device)
model.load_state_dict(torch.load("final_global_model.pth", map_location=device))
model.eval()
print("Pre-trained model loaded. Preparing to compute predictions!")


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
    
    num_augmentations = random.choice([1,2])
    
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

def compute_predictions(model, dataloader, device, sorted_hashes=None, pca_folder="PCA"):
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

# Load dataset indices
train_indices = torch.load("train_loader.pth")
test_indices = torch.load("test_loader.pth")

# Load dataset
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

train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=32, shuffle=False)
test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=32, shuffle=False)

# Compute predictions
sorted_hashes = np.load("sorted_train_phashes_decimal.npy")

train_results = compute_predictions(model.to(device), train_loader, device, sorted_hashes)
test_results = compute_predictions(model.to(device), test_loader, device, sorted_hashes)


# Save prediction results
torch.save(train_results, "target_mix_train_res.pt")
torch.save(test_results, "target_mix_test_res.pt")

print("Prediction results saved as target_mix_train_res.pt and target_mix_test_res.pt")
