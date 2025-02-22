import os
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import pickle

# Load partitioned CIFAR-100 dataset
partition_file = 'cifar100_partition1.pkl'
with open(partition_file, 'rb') as f:
    partition_data = pickle.load(f)

x_train = partition_data['train_data']
y_train = partition_data['train_labels']

# Normalize images to range [0, 1]
x_train = x_train.astype("float32") / 255.0

# Create PCA folder if it doesn't exist
output_folder = "PCA"
os.makedirs(output_folder, exist_ok=True)

# Iterate through all 100 classes
for class_label in range(100):
    # Extract all images of the current class
    class_images = x_train[y_train.flatten() == class_label]

    if len(class_images) == 0:
        print(f"No images found for class {class_label}, skipping...")
        continue

    # Reshape each image into a 1D array
    num_images, img_height, img_width, num_channels = class_images.shape
    class_images_flat = class_images.reshape(num_images, -1)

    # Perform PCA
    pca = PCA(n_components=4)  # Only retain the wanted principal component
    pca_class = pca.fit_transform(class_images_flat)

    # Reconstruct the composite image using the first principal component
    mean_image_flat = pca.components_[0]  # Use the first principal component
    mean_image_flat = mean_image_flat * np.std(class_images_flat, axis=0) + np.mean(class_images_flat, axis=0)

    # Reshape back to the original image dimensions
    mean_image = mean_image_flat.reshape(img_height, img_width, num_channels)

    # Convert image from [0, 1] range to [0, 255] and save
    mean_image_uint8 = (mean_image * 255).astype("uint8")
    output_image = Image.fromarray(mean_image_uint8)

    # Save the output image to the PCA folder
    output_image_path = os.path.join(output_folder, f"cifar100_{class_label}_pca_composite.png")
    output_image.save(output_image_path)

    print(f"PCA composite image for class {class_label} saved as '{output_image_path}'")

