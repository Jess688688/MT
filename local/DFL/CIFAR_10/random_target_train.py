import torch

def perform_random_target_train(num_samples):
    shadow_train_res = torch.load("train_results.pt")

    shadow_train_logits, shadow_train_labels = shadow_train_res
    
    random_indices = torch.randint(0, shadow_train_logits.shape[0], (num_samples,))

    sampled_logits = shadow_train_logits[random_indices]
    sampled_labels = shadow_train_labels[random_indices]

    sampled_data = (sampled_logits, sampled_labels)
    
    torch.save(sampled_data, "random_target_train_res.pt")

    print("Sampled data saved successfully.")

if __name__ == "__main__":
    perform_random_target_train(5000)
