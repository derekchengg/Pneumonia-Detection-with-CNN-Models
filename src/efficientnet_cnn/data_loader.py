import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        categories = ["NORMAL", "PNEUMONIA"]
        for label, category in enumerate(categories):
            category_path = os.path.join(root_dir, category)
            if not os.path.exists(category_path):
                print(f"Warning: Missing data folder {category_path}")
                continue
            
            for img_name in os.listdir(category_path):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.image_paths.append(os.path.join(category_path, img_name))
                    self.labels.append(label)

        print(f"Loaded {len(self.image_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros((3, 224, 224)), label  # Placeholder for failed images


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    train_loader = DataLoader(PneumoniaDataset(os.path.join(data_dir, "train"), get_train_transform()),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(PneumoniaDataset(os.path.join(data_dir, "val"), get_test_transform()),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(PneumoniaDataset(os.path.join(data_dir, "test"), get_test_transform()),
                             batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader