import torch
from torch import nn
import json
import os
import torch
import lightning
from lightning import Trainer
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SoftmaxMLPClassifier(lightning.LightningModule):
    def __init__(self, input_dim, hidden_dim, learning_rate=0.001):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.long())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.long())
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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
        shadow_test_res = torch.load("shadow_test_res.pt")

        shadow_train_pre = shadow_train_res[0]
        shadow_test_pre = shadow_test_res[0]

        in_labels = torch.ones(shadow_train_pre.shape[0], dtype=torch.long)
        out_labels = torch.zeros(shadow_test_pre.shape[0], dtype=torch.long)

        attack_dataset = TensorDataset(torch.cat((shadow_train_pre, shadow_test_pre), dim=0),
                                        torch.cat((in_labels, out_labels), dim=0))

        attack_dataloader = DataLoader(attack_dataset, batch_size=128, shuffle=True, num_workers=0)

        attack_model = SoftmaxMLPClassifier(10, 64)

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

        in_eval_pre = torch.load("train_results.pt")
        out_eval_pre = torch.load("test_results.pt")

        in_predictions = in_out_samples_check(attack_model.to(self.device), in_eval_pre)
        out_predictions = in_out_samples_check(attack_model.to(self.device), out_eval_pre)

        true_positives = in_predictions.sum().item()
        false_positives = out_predictions.sum().item()

        precision, recall, f1 = self.evaluate_metrics(true_positives, false_positives, in_predictions)

        cifar10_train_accuracy = self.calculate_accuracy(in_eval_pre[0], in_eval_pre[1])
        cifar10_test_accuracy = self.calculate_accuracy(out_eval_pre[0], out_eval_pre[1])

        print(f"cifar10 Training Accuracy: {cifar10_train_accuracy:.2f}%")
        print(f"cifar10 Testing Accuracy: {cifar10_test_accuracy:.2f}%")

        return precision, recall, f1

def perform_shadow_model_mia():
    set_random_seed(42)
    attack_model = AttackModel()
    precision, recall, f1 = attack_model.MIA_shadow_model_attack()
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return precision, recall, f1

if __name__ == "__main__":
    perform_shadow_model_mia()
    