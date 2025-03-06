import random
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms, models
from pytorch_lightning import Trainer, LightningModule
import pickle
from PIL import Image
import torch.nn as nn
import gc
import torchmetrics
import torch.optim as optim

class TinyImageNet(LightningModule):
    def __init__(self, out_channels=200, learning_rate=1e-4):
        super(TinyImageNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, out_channels)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.learning_rate = learning_rate
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        acc = self.train_acc(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def _compute_predictions(model, dataloader, device):
    model = model.to(device)
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs, label = inputs.to(device), label.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs)
            labels.append(label)

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    return predictions, labels

def generate_shadow_datasets(num_shadow, train_data, test_data, train_size=10000, test_size=1000):
    shadow_train, shadow_test = [], []

    for _ in range(num_shadow):
        train_indices = random.sample(range(len(train_data)), train_size)
        test_indices = random.sample(range(len(test_data)), test_size)
        shadow_train.append(DataLoader(Subset(train_data, train_indices), batch_size=16, shuffle=True, num_workers=4))
        shadow_test.append(DataLoader(Subset(test_data, test_indices), batch_size=16, shuffle=False, num_workers=4))
    return shadow_train, shadow_test

def _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device, max_epochs=20):
    s_tr_pre, s_tr_label = [], []
    s_te_pre, s_te_label = [], []

    for i in range(num_shadow):
        print(f"Training Shadow Model {i+1}/{num_shadow}")

        shadow_model = TinyImageNet().to(device)
        shadow_trainer = Trainer(max_epochs=max_epochs, accelerator="gpu", devices=1, strategy="auto", logger=False, enable_checkpointing=False)
        shadow_trainer.fit(shadow_model, shadow_train[i])

        tr_pre, tr_label = _compute_predictions(shadow_model, shadow_train[i], device)
        te_pre, te_label = _compute_predictions(shadow_model, shadow_test[i], device)

        s_tr_pre.append(tr_pre)
        s_tr_label.append(tr_label)
        s_te_pre.append(te_pre)
        s_te_label.append(te_label)

        if hasattr(shadow_model, 'criterion'):
            del shadow_model.criterion
        del shadow_model
        torch.cuda.empty_cache()
        gc.collect()

    shadow_train_res = (torch.cat(s_tr_pre, dim=0), torch.cat(s_tr_label, dim=0))
    shadow_test_res = (torch.cat(s_te_pre, dim=0), torch.cat(s_te_label, dim=0))

    return shadow_train_res, shadow_test_res

def load_partitioned_tiny_imagenet(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = data['train_data'], data['train_labels']
    x_test, y_test = data['test_data'], data['test_labels']
    return x_train, y_train, x_test, y_test

partition_file = 'tiny_imagenet_partition2.pkl'
x_train, y_train, x_test, y_test = load_partitioned_tiny_imagenet(partition_file)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

x_train = torch.stack([transform(Image.fromarray(img)) for img in x_train])
y_train = torch.tensor(y_train).squeeze().long()

x_test = torch.stack([transform(Image.fromarray(img)) for img in x_test])
y_test = torch.tensor(y_test).squeeze().long()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

num_shadow = 10
shadow_train, shadow_test = generate_shadow_datasets(num_shadow, train_dataset, test_dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyImageNet().to(device)

shadow_train_res, shadow_test_res = _generate_attack_dataset(model, shadow_train, shadow_test, num_shadow, device)

print("Shadow Training Results:", shadow_train_res)
print("Shadow Testing Results:", shadow_test_res)

torch.save(shadow_train_res, "shadow_train_res_tiny_imagenet.pt")
torch.save(shadow_test_res, "shadow_test_res_tiny_imagenet.pt")

print("Shadow training results saved to shadow_train_res_tiny_imagenet.pt")
print("Shadow testing results saved to shadow_test_res_tiny_imagenet.pt")
