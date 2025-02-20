import numpy as np
import torch

class ClassMetricBasedAttack:
    def __init__(self, shadow_train_res_path, shadow_test_res_path, in_eval_pre_path, out_eval_pre_path):
        # Load data from .pt files
        self.shadow_train_res = torch.load(shadow_train_res_path)
        self.shadow_test_res = torch.load(shadow_test_res_path)
        self.in_eval_pre = torch.load(in_eval_pre_path)
        self.out_eval_pre = torch.load(out_eval_pre_path)

        # Extract number of classes
        self.num_classes = 10

        # Extract Shadow model outputs and labels
        self.s_in_outputs, self.s_in_labels = self.shadow_train_res
        self.s_out_outputs, self.s_out_labels = self.shadow_test_res

        # Extract Target model outputs and labels
        self.t_in_outputs, self.t_in_labels = self.in_eval_pre
        self.t_out_outputs, self.t_out_labels = self.out_eval_pre

        # Convert to NumPy arrays and ensure labels are 1D
        self.s_in_outputs = self.s_in_outputs.cpu().detach().numpy()
        self.s_in_labels = self.s_in_labels.cpu().detach().numpy().flatten()
        self.s_out_outputs = self.s_out_outputs.cpu().detach().numpy()
        self.s_out_labels = self.s_out_labels.cpu().detach().numpy().flatten()
        self.t_in_outputs = self.t_in_outputs.cpu().detach().numpy()
        self.t_in_labels = self.t_in_labels.cpu().detach().numpy().flatten()
        self.t_out_outputs = self.t_out_outputs.cpu().detach().numpy()
        self.t_out_labels = self.t_out_labels.cpu().detach().numpy().flatten()

        # Ensure outputs are 1D where necessary
        self.s_in_conf = np.array([self.s_in_outputs[i, self.s_in_labels[i]] for i in range(len(self.s_in_labels))])
        self.s_out_conf = np.array([self.s_out_outputs[i, self.s_out_labels[i]] for i in range(len(self.s_out_labels))])
        self.t_in_conf = np.array([self.t_in_outputs[i, self.t_in_labels[i]] for i in range(len(self.t_in_labels))])
        self.t_out_conf = np.array([self.t_out_outputs[i, self.t_out_labels[i]] for i in range(len(self.t_out_labels))])

        # Calculate entropy
        self.s_in_entr = self._entr_comp(self.s_in_outputs)
        self.s_out_entr = self._entr_comp(self.s_out_outputs)
        self.t_in_entr = self._entr_comp(self.t_in_outputs)
        self.t_out_entr = self._entr_comp(self.t_out_outputs)

        # Calculate modified entropy
        self.s_in_m_entr = self._m_entr_comp(self.s_in_outputs, self.s_in_labels)
        self.s_out_m_entr = self._m_entr_comp(self.s_out_outputs, self.s_out_labels)
        self.t_in_m_entr = self._m_entr_comp(self.t_in_outputs, self.t_in_labels)
        self.t_out_m_entr = self._m_entr_comp(self.t_out_outputs, self.t_out_labels)

    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / len(tr_values)
            te_ratio = np.sum(te_values < value) / len(te_values)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_thre(self, s_tr_values, s_te_values, t_tr_values, t_te_values):
        true_positives, false_positives = 0, 0

        for num in range(self.num_classes):
            thre = self._thre_setting(
                s_tr_values[self.s_in_labels == num], s_te_values[self.s_out_labels == num]
            )
            true_positives += np.sum(t_tr_values[self.t_in_labels == num] >= thre)
            false_positives += np.sum(t_te_values[self.t_out_labels == num] >= thre)

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (len(t_tr_values) + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return precision, recall, f1

    def mem_inf_benchmarks(self):
        results = {}

        # Prediction Class Confidence
        results["Prediction Class Confidence"] = self._mem_inf_thre(
            self.s_in_conf, self.s_out_conf, self.t_in_conf, self.t_out_conf
        )

        # Prediction Class Entropy
        results["Prediction Class Entropy"] = self._mem_inf_thre(
            -self.s_in_entr, -self.s_out_entr, -self.t_in_entr, -self.t_out_entr
        )

        # Prediction Modified Entropy
        results["Prediction Modified Entropy"] = self._mem_inf_thre(
            -self.s_in_m_entr, -self.s_out_m_entr, -self.t_in_m_entr, -self.t_out_m_entr
        )

        return results

if __name__ == "__main__":
    # File paths
    shadow_train_res_path = "shadow_train_res.pt"
    shadow_test_res_path = "shadow_test_res.pt"
    in_eval_pre_path = "train_results.pt"
    out_eval_pre_path = "test_results.pt"

    # Initialize the attack class
    attack = ClassMetricBasedAttack(
        shadow_train_res_path, shadow_test_res_path, in_eval_pre_path, out_eval_pre_path
    )

    # Get benchmark results
    benchmarks = attack.mem_inf_benchmarks()

    # Print results
    for method, metrics in benchmarks.items():
        precision, recall, f1 = metrics
        print(f"{method}: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")