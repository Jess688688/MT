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
import torchmetrics

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CIFAR10Model().to(device)
model.load_state_dict(torch.load("final_global_model.pth", map_location=device))
model.eval()
print("Pre-trained model is loaded. Prepare to predict!")

def apply_composite(img_array, pca_folder="PCA", alpha=0.5):
    pca_images = {}
    for class_id in range(10):
        pca_path = os.path.join(pca_folder, f"cifar10_{class_id}_pca_composite.png")
        pca_images[class_id] = np.array(Image.open(pca_path))
    
    pca_image = pca_images[random.choice(range(10))]
    pca_image = np.resize(pca_image, img_array.shape)
    
    fused_image = alpha * img_array + (1 - alpha) * pca_image
    fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
    
    return fused_image

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
    # num_augmentations = random.choice([0, 1])
    num_augmentations = 2
    # num_augmentations = random.choices([0, 1], weights=[0.5, 0.5])[0]
    
    selected_augmentations = transforms.Compose(random.sample(augmentations, num_augmentations))
    return np.array(selected_augmentations(image))

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
def calculate_phash_decimal(img_array):
    pil_image = Image.fromarray(img_array)
    phash_hex = str(imagehash.phash(pil_image))  # Calculate pHash in hex
    return int(phash_hex, 16)  # Convert to decimal

def compute_predictions(model, dataloader, device, sorted_hashes, pca_folder="PCA"):
    """Processes images before feeding them into the model."""
    model.eval()
    predictions, labels = [], []
    tempsum = 0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    ])

    with torch.no_grad():
        for inputs, label in dataloader:
            augmented_inputs = []
            for img_array, lbl in zip(inputs.numpy(), label):
                img_array = img_array.astype(np.uint8)
                phash_decimal = np.uint64(calculate_phash_decimal(img_array))
                
                if sorted_hashes is not None and sorted_hashes.size > 0 and binary_search(sorted_hashes, phash_decimal):
                    image = Image.fromarray(img_array)
                    img_array = apply_random_augmentation(image)
                    img_array = apply_composite(img_array, pca_folder=pca_folder)
                    tempsum += 1
                
                img_tensor = transform(Image.fromarray(img_array))  # Convert back to tensor after augmentation
                augmented_inputs.append(img_tensor)
            
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

train_dataset = list(zip(x_train, y_train))
test_dataset = list(zip(x_test, y_test))

train_loaders = [DataLoader(Subset(train_dataset, indices), batch_size=16, shuffle=False) for indices in train_indices_list]
test_loaders = [DataLoader(Subset(test_dataset, indices), batch_size=16, shuffle=False) for indices in test_indices_list]

def generate_sorted_train_phashes(dataset):
    phashes = []
    for img_array, _ in dataset:
        img_array = img_array.astype(np.uint8)  # Ensure image remains in uint8 format
        phash_decimal = calculate_phash_decimal(img_array)
        phashes.append(phash_decimal)
    
    sorted_phashes = np.sort(phashes)
    np.save("sorted_train_phashes_decimal.npy", sorted_phashes)
    print("Sorted pHash values saved to sorted_train_phashes_decimal.npy")
    return sorted_phashes

sorted_hashes = generate_sorted_train_phashes(train_dataset)

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

train_results = (torch.cat(final_train_predictions, dim=0), torch.cat(final_train_labels, dim=0))
test_results = (torch.cat(final_test_predictions, dim=0), torch.cat(final_test_labels, dim=0))

torch.save(train_results, "train_results.pt")
torch.save(test_results, "test_results.pt")

print("Prediction results are saved!")
