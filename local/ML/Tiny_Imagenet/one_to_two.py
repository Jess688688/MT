import numpy as np
import pickle
import os
import urllib.request
import zipfile
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def download_and_extract_tiny_imagenet(dataset_path="tiny-imagenet-200"):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = "tiny-imagenet-200.zip"
    
    if not os.path.exists(dataset_path):
        print("Downloading Tiny ImageNet dataset...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Dataset downloaded and extracted.")
        os.remove(zip_path)
    else:
        print("Tiny ImageNet dataset already exists.")

download_and_extract_tiny_imagenet()

TINY_IMAGENET_PATH = "tiny-imagenet-200"
TRAIN_PATH = os.path.join(TINY_IMAGENET_PATH, "train")
VAL_PATH = os.path.join(TINY_IMAGENET_PATH, "val")

class_names = sorted(os.listdir(TRAIN_PATH))
num_classes = len(class_names)

class_to_index = {cls: i for i, cls in enumerate(class_names)}

def load_tiny_imagenet_data():
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    
    for cls in class_names:
        class_path = os.path.join(TRAIN_PATH, cls, "images")
        image_files = sorted(os.listdir(class_path))
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            img = load_img(img_path, target_size=(64, 64))  # 确保大小为 64x64
            img_array = img_to_array(img)
            train_data.append(img_array)
            train_labels.append(class_to_index[cls])
    
    with open(os.path.join(VAL_PATH, "val_annotations.txt"), "r") as f:
        val_annotations = f.readlines()
    val_img_to_label = {line.split()[0]: line.split()[1] for line in val_annotations}
    
    val_images_path = os.path.join(VAL_PATH, "images")
    for img_file in sorted(os.listdir(val_images_path)):
        img_path = os.path.join(val_images_path, img_file)
        img = load_img(img_path, target_size=(64, 64))
        img_array = img_to_array(img)
        test_data.append(img_array)
        test_labels.append(class_to_index[val_img_to_label[img_file]])
    
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

x_train, y_train, x_test, y_test = load_tiny_imagenet_data()

train_class_indices = {i: np.where(y_train == i)[0] for i in range(num_classes)}
test_class_indices = {i: np.where(y_test == i)[0] for i in range(num_classes)}

train1_indices, train2_indices = [], []
test1_indices, test2_indices = [], []

for i in range(num_classes):
    train_indices = train_class_indices[i]
    np.random.shuffle(train_indices)
    train_split = len(train_indices) // 2
    train1_indices.extend(train_indices[:train_split])
    train2_indices.extend(train_indices[train_split:])
    
    test_indices = test_class_indices[i]
    np.random.shuffle(test_indices)
    test_split = len(test_indices) // 2
    test1_indices.extend(test_indices[:test_split])
    test2_indices.extend(test_indices[test_split:])

train1_indices, train2_indices = np.array(train1_indices), np.array(train2_indices)
test1_indices, test2_indices = np.array(test1_indices), np.array(test2_indices)

x_train1, y_train1 = x_train[train1_indices], y_train[train1_indices]
x_train2, y_train2 = x_train[train2_indices], y_train[train2_indices]
x_test1, y_test1 = x_test[test1_indices], y_test[test1_indices]
x_test2, y_test2 = x_test[test2_indices], y_test[test2_indices]

x_train1 = x_train1.astype(np.uint8)
x_train2 = x_train2.astype(np.uint8)
x_test1 = x_test1.astype(np.uint8)
x_test2 = x_test2.astype(np.uint8)

y_train1 = y_train1.astype(np.uint8)
y_train2 = y_train2.astype(np.uint8)
y_test1 = y_test1.astype(np.uint8)
y_test2 = y_test2.astype(np.uint8)

def save_tiny_imagenet_partition(train_data, train_labels, test_data, test_labels, filename):
    with open(filename, 'wb') as f:
        pickle.dump({
            'train_data': train_data,
            'train_labels': train_labels,
            'test_data': test_data,
            'test_labels': test_labels
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

y_train1 = y_train1.reshape(-1, 1)  
y_train2 = y_train2.reshape(-1, 1)  
y_test1 = y_test1.reshape(-1, 1)  
y_test2 = y_test2.reshape(-1, 1)  

save_tiny_imagenet_partition(x_train1, y_train1, x_test1, y_test1, 'tiny_imagenet_partition1.pkl')
save_tiny_imagenet_partition(x_train2, y_train2, x_test2, y_test2, 'tiny_imagenet_partition2.pkl')

print("Tiny ImageNet is divided and saved as 'tiny_imagenet_partition1.pkl' 和 'tiny_imagenet_partition2.pkl'")
