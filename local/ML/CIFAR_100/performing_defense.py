import torch
import torch.nn as nn
from torchvision import transforms
import pickle
import random
import os
import numpy as np
from PIL import Image
import imagehash
from sklearn.decomposition import PCA
from torchvision import transforms, models
from pytorch_lightning import LightningModule

# Load Model
class CIFAR100Model(LightningModule):
    def __init__(self, num_classes=100):
        super(CIFAR100Model, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

def calculate_phash_decimal(image):
    pil_image = Image.fromarray(image)
    phash_hex = str(imagehash.phash(pil_image))  # Calculate pHash in hex
    return int(phash_hex, 16)  # Convert to decimal

def preload_pca_images(pca_folder="PCA"):
    pca_images = {}
    for class_id in range(100):
        pca_path = os.path.join(pca_folder, f"cifar100_{class_id}_pca_composite.png")
        if os.path.exists(pca_path):
            pca_images[class_id] = np.array(Image.open(pca_path))
    return pca_images

pca_images = preload_pca_images()

def apply_composite(img, pca_images, alpha=0.5):
    img_array = np.array(img)
    random_class = np.random.randint(0, 100)
    pca_image = pca_images[random_class]    
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
    num_augmentations = random.choices([0, 1], weights=[0.5, 0.5])[0]
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
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])
    
    tempsum = 0
    with torch.no_grad():
        augmented_images = []
        for img, lbl in zip(raw_images, labels):
            phash_decimal = calculate_phash_decimal(img)
            img = Image.fromarray(img)
            
            if sorted_hashes is not None and binary_search(sorted_hashes, phash_decimal):
                img = apply_random_augmentation(img)
                img = apply_composite(img, pca_images)
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
with open("cifar100_partition1.pkl", 'rb') as f:
    data = pickle.load(f)

raw_x_train, y_train = data['train_data'], data['train_labels']
raw_x_test, y_test = data['test_data'], data['test_labels']

# load pHash list
sorted_hashes = np.load("sorted_train_phashes_decimal.npy")

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR100Model().to(device)
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
