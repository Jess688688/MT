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

class ImageNet10(LightningModule):
    def __init__(self, out_channels=10, learning_rate=1e-3):
        super(ImageNet10, self).__init__()
        self.learning_rate = learning_rate
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)

    def forward(self, x):
        return self.model(x)

def preload_pca_images(pca_folder="PCA"):
    pca_images = {}
    for class_id in range(10):
        pca_path = os.path.join(pca_folder, f"imagenet10_{class_id}_pca_composite.png")
        if os.path.exists(pca_path):
            pca_images[class_id] = np.array(Image.open(pca_path))
    return pca_images

def apply_composite(img, pca_images, alpha):
    img_array = np.array(img)
    random_class = np.random.randint(0, 10)
    pca_image = pca_images[random_class]    
    pca_image = np.resize(pca_image, img_array.shape)
    fused_image = alpha * img_array + (1 - alpha) * pca_image
    fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
    return Image.fromarray(fused_image)

def apply_random_augmentation(image, num, weights):
    augmentations = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=(20, 22)),
        transforms.RandomAffine(degrees=(10, 12), translate=(0.1, 0.1)),
        transforms.RandomAffine(degrees=(3, 5), scale=(0.95, 0.95)),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 0.9)),
        transforms.RandomPerspective(distortion_scale=0.08, p=1),
        transforms.RandomEqualize(p=1),
        transforms.RandomCrop(size=(224, 224), padding=4),
        transforms.GaussianBlur(kernel_size=(21, 21), sigma=(6, 6)),
        transforms.RandomGrayscale(p=1.0),
        transforms.RandomAdjustSharpness(sharpness_factor=4, p=1),
        transforms.RandomPosterize(bits=4, p=1),
    ]
    num_augmentations = random.choices(num, weights)[0]
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

def calculate_phash_decimal(img_array):
    pil_image = Image.fromarray(img_array)
    phash_hex = str(imagehash.phash(pil_image))
    return int(phash_hex, 16)

def compute_predictions(model, raw_images, labels, device, sorted_hashes, pca_images, num, weights, alpha):
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
                img = apply_random_augmentation(img, num, weights)
                img = apply_composite(img, pca_images, alpha)
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

def perform_defense(num, weights, alpha):  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageNet10().to(device)
    model.load_state_dict(torch.load("final_global_model.pth", map_location=device))
    model.eval()
    print("Pre-trained model is loaded. Prepare to predict!")

    pca_images = preload_pca_images()

    sorted_hashes = np.load("sorted_train_phashes_decimal.npy")

    train_indices_list = torch.load("train_loader.pth")
    test_indices_list = torch.load("test_loader.pth")

    with open("imagenet10_partition1.pkl", 'rb') as f:
        data = pickle.load(f)

    x_train, y_train = data['train_data'], data['train_labels']
    x_test, y_test = data['test_data'], data['test_labels']
        
    train_dataset = list(zip(x_train, y_train))
    test_dataset = list(zip(x_test, y_test))

    train_loaders = [DataLoader(Subset(train_dataset, indices), batch_size=16, shuffle=False) for indices in train_indices_list]
    test_loaders = [DataLoader(Subset(test_dataset, indices), batch_size=16, shuffle=False) for indices in test_indices_list]

    final_train_predictions, final_train_labels = [], []
    final_test_predictions, final_test_labels = [], []

    for i, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
        print(f"Computing predictions for participant {i+1}")

        x_train_extracted = np.concatenate([x.cpu().detach().numpy() for x, _ in train_loader], axis=0)
        y_train_extracted = np.concatenate([y.cpu().detach().numpy() for _, y in train_loader], axis=0)

        x_test_extracted = np.concatenate([x.cpu().detach().numpy() for x, _ in test_loader], axis=0)
        y_test_extracted = np.concatenate([y.cpu().detach().numpy() for _, y in test_loader], axis=0)
        
        train_results = compute_predictions(model, x_train_extracted, y_train_extracted, device, sorted_hashes, pca_images, num, weights, alpha)
        test_results = compute_predictions(model, x_test_extracted, y_test_extracted, device, sorted_hashes, pca_images, num, weights, alpha)

        # train_results = compute_predictions(model, x_train, y_train, device, sorted_hashes, pca_images, num, weights, alpha)
        # test_results = compute_predictions(model, x_test, y_test, device, sorted_hashes, pca_images, num, weights, alpha)

        final_train_predictions.append(train_results[0])
        final_train_labels.append(train_results[1])
        final_test_predictions.append(test_results[0])
        final_test_labels.append(test_results[1])

    train_results = (torch.cat(final_train_predictions, dim=0), torch.cat(final_train_labels, dim=0))
    test_results = (torch.cat(final_test_predictions, dim=0), torch.cat(final_test_labels, dim=0))

    torch.save(train_results, "train_results.pt")
    torch.save(test_results, "test_results.pt")

    print("Prediction results are saved!")

if __name__ == "__main__":
    perform_defense([0, 1], [0.5, 0.5], 0.5)
