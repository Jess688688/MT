import torch
from torch import nn
import json
import os
import torch
import lightning
from lightning import Trainer
import torch.nn.functional as F
import lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import random
import numpy as np

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BinaryClassifier(pl.LightningModule):
    def __init__(self, input_dim=10, num_filters=64, kernel_size=3, learning_rate=0.001):
        super().__init__()
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size, padding=1)
        self.bn3 = nn.BatchNorm1d(num_filters * 4)
        
        self.fc1 = nn.Linear((num_filters * 4) * input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

        self.dropout = nn.Dropout(0.3)
        self.learning_rate = learning_rate

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.long())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}

class AttackModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def evaluate_metrics(self, tp, fp, in_predictions):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(in_predictions) if len(in_predictions) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def calculate_accuracy(self, predictions, labels):
        labels = labels.view(-1) 

        predicted_classes = torch.argmax(predictions, dim=1)
        correct = (predicted_classes == labels).sum().item()
        total_samples = labels.size(0)

        if total_samples == 0:
            return 0

        accuracy = correct / total_samples * 100

        return accuracy

    def MIA_shadow_model_attack(self):

        shadow_train_res = torch.load("random_shadow_train_res.pt")
        shadow_test_res = torch.load("s_test_results.pt")

        shadow_train_pre = shadow_train_res[0]
        shadow_test_pre = shadow_test_res[0]

        in_labels = torch.ones(shadow_train_pre.shape[0], dtype=torch.long)
        out_labels = torch.zeros(shadow_test_pre.shape[0], dtype=torch.long)

        attack_dataset = TensorDataset(torch.cat((shadow_train_pre, shadow_test_pre), dim=0),
                                        torch.cat((in_labels, out_labels), dim=0))

        attack_dataloader = DataLoader(attack_dataset, batch_size=128, shuffle=True, num_workers=0)

        attack_model = BinaryClassifier(10, 64)

        attack_trainer = Trainer(max_epochs=50, accelerator="auto", devices="auto", logger=False,
                                    enable_checkpointing=False, enable_model_summary=False)
        attack_trainer.fit(attack_model, attack_dataloader)

        def in_out_samples_check(model, dataset):
            predictions, _ = dataset
            dataloader = DataLoader(predictions, batch_size=128, shuffle=False, num_workers=0)

            predicted_label = []
            model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    logits = model(batch)
                    _, predicted = torch.max(logits, 1)

                    true_items = predicted == 1
                    predicted_label.append(true_items)

                predicted_label = torch.cat(predicted_label, dim=0)
            return predicted_label

        in_eval_pre = torch.load("random_target_train_res.pt")
        out_eval_pre = torch.load("test_results.pt")

        in_predictions = in_out_samples_check(attack_model.to(self.device), in_eval_pre)
        out_predictions = in_out_samples_check(attack_model.to(self.device), out_eval_pre)

        true_positives = in_predictions.sum().item()
        false_positives = out_predictions.sum().item()

        precision, recall, f1 = self.evaluate_metrics(true_positives, false_positives, in_predictions)

        cifar10_train_accuracy = self.calculate_accuracy(in_eval_pre[0], in_eval_pre[1])
        cifar10_test_accuracy = self.calculate_accuracy(out_eval_pre[0], out_eval_pre[1])

        print(f"Imagenet-10 Training Accuracy: {cifar10_train_accuracy:.2f}%")
        print(f"Imagenet-10 Testing Accuracy: {cifar10_test_accuracy:.2f}%")

        return precision, recall, f1

def perform_shadow_model_mia():
    set_random_seed(42)
    attack_model = AttackModel()
    precision, recall, f1 = attack_model.MIA_shadow_model_attack()
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return precision, recall, f1

if __name__ == "__main__":
    perform_shadow_model_mia()