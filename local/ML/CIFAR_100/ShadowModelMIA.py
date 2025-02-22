import torch
from torch import nn
import json
import os
import torch
import lightning
from lightning import Trainer
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class SoftmaxMLPClassifier(lightning.LightningModule):
    def __init__(self, input_dim, hidden_dim, learning_rate=0.001):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Additional hidden layer
        self.fc3 = nn.Linear(hidden_dim, 2)  # Output layer with 2 units for softmax
        self.learning_rate = learning_rate

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Activation for additional hidden layer
        x = self.fc3(x)  # No sigmoid activation, logits are expected by CrossEntropyLoss
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.long())  # Use cross_entropy, which includes softmax
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.long())  # Use cross_entropy, which includes softmax
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class AttackModel:
    def __init__(self):
        # 添加设备属性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_metrics(self, tp, fp, in_predictions):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(in_predictions) if len(in_predictions) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def calculate_accuracy(self, predictions, labels):
        """
        Calculate accuracy for the given predictions and labels.
        """
        predicted_classes = torch.argmax(predictions, dim=1)
        correct = (predicted_classes == labels).sum().item()
        accuracy = correct / labels.size(0) * 100  # Convert to percentage
        return accuracy

    def MIA_shadow_model_attack(self):

        shadow_train_res = torch.load("shadow_train_res_cifar100.pt")
        shadow_test_res = torch.load("shadow_test_res_cifar100.pt")

        shadow_train_pre = shadow_train_res[0]
        shadow_test_pre = shadow_test_res[0]

        in_labels = torch.ones(shadow_train_pre.shape[0], dtype=torch.long)
        out_labels = torch.zeros(shadow_test_pre.shape[0], dtype=torch.long)

        attack_dataset = TensorDataset(torch.cat((shadow_train_pre, shadow_test_pre), dim=0),
                                        torch.cat((in_labels, out_labels), dim=0))

        attack_dataloader = DataLoader(attack_dataset, batch_size=128, shuffle=True, num_workers=0)

        # CIFAR-100 has 100 classes, so input_dim should be 100
        attack_model = SoftmaxMLPClassifier(100, 64)

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
                    _, predicted = torch.max(logits, 1)  # max value, max value index

                    true_items = predicted == 1
                    predicted_label.append(true_items)

                predicted_label = torch.cat(predicted_label, dim=0)
            return predicted_label

        in_eval_pre = torch.load("target_train_res.pt")
        out_eval_pre = torch.load("target_test_res.pt")

        in_predictions = in_out_samples_check(attack_model.to(self.device), in_eval_pre)
        out_predictions = in_out_samples_check(attack_model.to(self.device), out_eval_pre)

        true_positives = in_predictions.sum().item()
        false_positives = out_predictions.sum().item()

        precision, recall, f1 = self.evaluate_metrics(true_positives, false_positives, in_predictions)

        # Calculate accuracy for target training and testing datasets
        cifar100_train_accuracy = self.calculate_accuracy(in_eval_pre[0], in_eval_pre[1])
        cifar100_test_accuracy = self.calculate_accuracy(out_eval_pre[0], out_eval_pre[1])

        print(f"CIFAR-100 Training Accuracy: {cifar100_train_accuracy:.2f}%")
        print(f"CIFAR-100 Testing Accuracy: {cifar100_test_accuracy:.2f}%")

        return precision, recall, f1


if __name__ == "__main__":
    attack_model = AttackModel()
    precision, recall, f1 = attack_model.MIA_shadow_model_attack()
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
