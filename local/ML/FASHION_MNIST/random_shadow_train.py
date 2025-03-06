import torch

shadow_train_res = torch.load("shadow_train_res.pt")

if isinstance(shadow_train_res, tuple):
    shadow_train_logits, shadow_train_labels = shadow_train_res

    num_samples = 5000

    random_indices = torch.randint(0, shadow_train_logits.shape[0], (num_samples,))

    sampled_logits = shadow_train_logits[random_indices]
    sampled_labels = shadow_train_labels[random_indices]

    sampled_data = (sampled_logits, sampled_labels)
    
    torch.save(sampled_data, "random_shadow_train_res.pt")

    print("Sampled data saved successfully.")
else:
    print("Unexpected Data Structure:", type(shadow_train_res))