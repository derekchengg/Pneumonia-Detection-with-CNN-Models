import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os


# Load the data and output a saved numpy file to avoid loading data everytime we train the CNN model

labels = ['PNEUMONIA', 'NORMAL']
img_size = 175

def load_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"error reading {img}: {e}")
    return data


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(BASE_DIR, "..", "..", "data", "train")
    test_path = os.path.join(BASE_DIR, "..", "..", "data", "test")
    val_path = os.path.join(BASE_DIR, "..", "..", "data", "val")
    
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    val_data = load_data(test_path)
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    x_val, y_val = [], []
    
    datasets = {
    'train': (train_data, x_train, y_train),
    'val': (val_data, x_val, y_val),
    'test': (test_data, x_test, y_test)
    }
    
    for name, (data, x_list, y_list) in datasets.items():
        for feature, label in data:
            x_list.append(feature)
            y_list.append(label)

    x_train = np.array(x_train).reshape(-1, img_size, img_size, 1) / 255.0
    y_train = np.array(y_train)

    x_val = np.array(x_val).reshape(-1, img_size, img_size, 1) / 255.0
    y_val = np.array(y_val)

    x_test = np.array(x_test).reshape(-1, img_size, img_size, 1) / 255.0
    y_test = np.array(y_test)
    
    output_dir = os.path.join(BASE_DIR, "..", "..", "processed_data")
    os.makedirs(output_dir, exist_ok=True)

    for name, (x_list, y_list) in {
        'train': (x_train, y_train),
        'val': (x_val, y_val),
        'test': (x_test, y_test)
        }.items():
        
        X = np.array(x_list).reshape(-1, img_size, img_size, 1) / 255.0
        y = np.array(y_list)

        np.save(os.path.join(output_dir, f'X_{name}.npy'), X)
        np.save(os.path.join(output_dir, f'y_{name}.npy'), y)

        print(f"Saved X_{name}.npy and y_{name}.npy to processed_data: shape: {X.shape}, labels: {y.shape}")




if __name__ == "__main__":
    main()
