import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
import pickle

# 加载模型
class CIFAR10Model(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10Model, self).__init__()
        self.model = models.vgg16()
        self.model.classifier[2] = nn.Dropout(0.5)
        self.model.classifier[5] = nn.Dropout(0.5)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CIFAR10Model().to(device)
model.load_state_dict(torch.load("final_global_model.pth", map_location=device))
model.eval()
print("Pre-trained model is loaded. Prepare to predict!")

# compute predictions`
def compute_predictions(model, dataloader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for inputs, label in dataloader:
            inputs, label = inputs.to(device), label.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs.cpu())
            labels.append(label.cpu())

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

train_results = compute_predictions(model, train_loaders[0], device)
test_results = compute_predictions(model, test_loaders[0], device)

torch.save(train_results, "train_results.pt")
torch.save(test_results, "test_results.pt")

print("prediction results are saved！")
