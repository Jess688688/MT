import os
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import pickle

partition_file = 'fashion_mnist_partition1.pkl'
with open(partition_file, 'rb') as f:
    partition_data = pickle.load(f)

x_train = partition_data['train_data']
y_train = partition_data['train_labels']

x_train = x_train.astype("float32") / 255.0

output_folder = "PCA"
os.makedirs(output_folder, exist_ok=True)

for class_label in range(10):
    class_images = x_train[y_train.flatten() == class_label]

    num_images, img_height, img_width = class_images.shape
    class_images_flat = class_images.reshape(num_images, -1)

    pca = PCA(n_components=4)
    pca_class = pca.fit_transform(class_images_flat)

    mean_image_flat = pca.components_[0]
    mean_image_flat = mean_image_flat * np.std(class_images_flat, axis=0) + np.mean(class_images_flat, axis=0)

    mean_image = mean_image_flat.reshape(img_height, img_width)

    mean_image_uint8 = (mean_image * 255).astype("uint8")
    output_image = Image.fromarray(mean_image_uint8, mode="L")

    output_image_path = os.path.join(output_folder, f"fashion_mnist_{class_label}_pca_composite.png")
    output_image.save(output_image_path)

    print(f"PCA composite image for class {class_label} saved as '{output_image_path}'")