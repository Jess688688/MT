import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms, models
import pickle
import random
import os
import numpy as np
from PIL import Image
import imagehash
from sklearn.decomposition import PCA
from pytorch_lightning import Trainer, LightningModule
import torchmetrics

class TinyImageNet(LightningModule):
    def __init__(self, out_channels=200, learning_rate=1e-4):
        super(TinyImageNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, out_channels)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)

    def forward(self, x):
        return self.model(x)

def calculate_phash_decimal(image):
    pil_image = Image.fromarray(image)
    phash_hex = str(imagehash.phash(pil_image))
    return int(phash_hex, 16)

def preload_pca_images(pca_folder="PCA"):
    pca_images = {}
    for class_id in range(200):
        pca_path = os.path.join(pca_folder, f"tiny_imagenet_{class_id}_pca_composite.png")
        if os.path.exists(pca_path):
            pca_images[class_id] = np.array(Image.open(pca_path))
    return pca_images

def apply_composite(img, pca_images, alpha):
    img_array = np.array(img)
    random_class = np.random.randint(0, 200)
    pca_image = pca_images[random_class]
    pca_image = np.resize(pca_image, img_array.shape)
    fused_image = alpha * img_array + (1 - alpha) * pca_image
    fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
    fused_pil_image = Image.fromarray(fused_image)
    return fused_pil_image

def apply_random_augmentation(image, num, weights):
    augmentations = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=(20, 22)),
        transforms.RandomAffine(degrees=(10, 12), translate=(0.1, 0.1)),
        transforms.RandomAffine(degrees=(3, 5), scale=(0.95, 0.95)),
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.9, 0.9)),
        transforms.RandomPerspective(distortion_scale=0.08, p=1),
        transforms.RandomEqualize(p=1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(2, 2)),
        transforms.RandomGrayscale(p=1.0),
        transforms.RandomAdjustSharpness(sharpness_factor=4, p=1),
        transforms.RandomPosterize(bits=4, p=1),
    ]
    
    num_augmentations = random.choices(num, weights)[0]
    selected_augmentations = transforms.Compose(random.sample(augmentations, num_augmentations))
    final_image = selected_augmentations(image)
    return final_image

def binary_search(sorted_arr, target):
    idx = np.searchsorted(sorted_arr, target)
    return idx < len(sorted_arr) and sorted_arr[idx] == target

def compute_predictions(model, raw_images, labels, device, sorted_hashes, pca_images, num, weights, alpha):
    model.eval()
    predictions, all_labels = [], []
    tempsum = 0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

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

    dataloader = DataLoader(TensorDataset(augmented_inputs, labels), batch_size=16, shuffle=False)

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs)
            all_labels.append(batch_y)
    
    print("tempsum equals:", tempsum)
    return torch.cat(predictions, dim=0), torch.cat(all_labels, dim=0)

def performance_defense(num, weights, alpha):
    pca_images = preload_pca_images()
    with open("random_query.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    loaded_x_train = loaded_data["train_data"]
    loaded_y_train = loaded_data["train_labels"]
    loaded_x_test = loaded_data["test_data"]
    loaded_y_test = loaded_data["test_labels"]
    sorted_hashes = np.load("sorted_train_phashes_decimal.npy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyImageNet().to(device)
    model.load_state_dict(torch.load("final_global_model_tiny_imagenet.pth", map_location=device))
    model.eval()
    print("Pre-trained model loaded. Computing predictions!")
    torch.cuda.empty_cache()
    train_results = compute_predictions(model, loaded_x_train, loaded_y_train, device, sorted_hashes, pca_images, num, weights, alpha)
    torch.cuda.empty_cache()
    test_results = compute_predictions(model, loaded_x_test, loaded_y_test, device, sorted_hashes, pca_images, num, weights, alpha)
    torch.cuda.empty_cache()
    torch.save(train_results, "train_results.pt")
    torch.save(test_results, "test_results.pt")

    print("Prediction results saved as train_results.pt and test_results.pt")

if __name__ == "__main__":
    performance_defense([0, 1], [1, 0], 0.8)
