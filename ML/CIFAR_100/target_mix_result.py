import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms, models, datasets
from pytorch_lightning import Trainer, LightningModule
import os
import pickle
import torchmetrics
import numpy as np
from PIL import Image
import imagehash

class CIFAR100Model(LightningModule):
    def __init__(self, num_classes=100):
        super(CIFAR100Model, self).__init__()
        self.model = models.vgg16(pretrained=True)
        
        # 训练整个模型，不冻结卷积层
        self.model.classifier[2] = nn.Dropout(0.5)
        self.model.classifier[5] = nn.Dropout(0.5)
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

def apply_composite(img, label, pca_folder="PCA", alpha=0.7):
    # Convert the input tensor to a numpy array
    img_array = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    # Load PCA composite images
    pca_images = {}
    for class_id in range(100):
        pca_path = os.path.join(pca_folder, f"cifar100_{class_id}_pca_composite.png")
        pca_images[class_id] = np.array(Image.open(pca_path))
    
    # Determine whether to use the same or a different class PCA image        
    pca_image = pca_images[random.choice(range(100))]  # 直接从 10 个类别中随机选

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
    
    # probabilities = [0.5, 0.5]
    num_augmentations = random.choice([1,2])
    
    # num_augmentations = 13
    selected_augmentations = random.sample(augmentations, num_augmentations)

    image = (image * 255).to(torch.uint8)

    for augment in selected_augmentations:
        image = augment(image)

    image = image.to(torch.float32) / 255.0

    return image

# Function to compute predictions
def _compute_predictions(model, dataloader, device, sorted_hashes=None, pca_folder="PCA"):
    model.eval()
    model.to(device)
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

# Generate Shadow Datasets
def generate_shadow_datasets(num_shadow, train_data, test_data, train_size=25000, test_size=5000):
    shadow_train, shadow_test = [], []
    
    for _ in range(num_shadow):
        train_indices = random.sample(range(len(train_data)), train_size)
        test_indices = random.sample(range(len(test_data)), test_size)

        shadow_train.append(DataLoader(Subset(train_data, train_indices), batch_size=32, shuffle=True))
        shadow_test.append(DataLoader(Subset(test_data, test_indices), batch_size=32, shuffle=False))
    
    return shadow_train, shadow_test

# Generate Attack Dataset
def _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device, sorted_hashes=None, max_epochs=50):
    s_tr_pre, s_tr_label = [], []
    s_te_pre, s_te_label = [], []
    
    for i in range(num_shadow):
        shadow_model = CIFAR100Model()
        shadow_model.to(device)
        
        shadow_trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices=1, logger=True, enable_checkpointing=False)
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


# Load partitioned CIFAR-100 dataset
def load_partitioned_cifar100(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = data['train_data'], data['train_labels']
    x_test, y_test = data['test_data'], data['test_labels']
    return x_train, y_train, x_test, y_test

# Replace CIFAR-10 with partitioned dataset
partition_file = 'cifar100_partition1.pkl'
x_train, y_train, x_test, y_test = load_partitioned_cifar100(partition_file)

# Convert to PyTorch Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

x_train = torch.tensor(x_train).permute(0, 3, 1, 2).float() / 255  # Convert to channels-first format
y_train = torch.tensor(y_train).squeeze().long()

x_test = torch.tensor(x_test).permute(0, 3, 1, 2).float() / 255
y_test = torch.tensor(y_test).squeeze().long()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

sorted_train_hashes = generate_sorted_train_phashes(train_dataset)

# Generate shadow datasets
num_shadow = 1
shadow_train, shadow_test = generate_shadow_datasets(num_shadow, train_dataset, test_dataset)

# Initialize the main model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR100Model().to(device)

# Generate attack dataset
target_train_res, target_test_res = _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device, sorted_train_hashes)

print("Target Training Results:", target_train_res)
print("Target Testing Results:", target_test_res)

# Save results
torch.save(target_train_res, "target_mix_train_res.pt")
torch.save(target_test_res, "target_mix_test_res.pt")

print("Target composite training results saved to target_mix_train_res.pt")
print("Target composite testing results saved to target_mix_test_res.pt")


