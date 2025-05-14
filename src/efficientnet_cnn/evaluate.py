import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from data_loader import PneumoniaDataset, get_test_transform
from efficientnet import get_efficientnet_model, load_model_from_checkpoint


def evaluate_model(model_path, test_dir, model_name="efficientnet_b2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load only test dataset
    test_dataset = PneumoniaDataset(test_dir, transform=get_test_transform())

    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty! Ensure 'data/chest_xray/test' has images in 'NORMAL' and 'PNEUMONIA'.")

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    model = get_efficientnet_model(model_name)
    model = load_model_from_checkpoint(model, model_path, device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = (probs[:, 1] > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Test Accuracy: {accuracy_score(all_labels, all_preds) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))
    print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))

    # --- Visualization 1: Class Distribution ---
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=all_labels, palette='coolwarm')
    ax.set_xticklabels(['Normal', 'Pneumonia'])
    plt.title("Class Distribution in Test Set", fontsize=14)
    plt.xlabel("Class")
    plt.ylabel("Count")

    total = len(all_labels)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height} ({height/total:.1%})', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()

    # --- Visualization 2: Confusion Matrix Heatmap ---
    cm = confusion_matrix(all_labels, all_preds)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)"

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=["Normal", "Pneumonia"],
                yticklabels=["Normal", "Pneumonia"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # --- Visualization 3: Random Sample of Predictions ---
    sample_indices = np.random.choice(len(test_dataset), 8, replace=False)
    images, labels = zip(*[test_dataset[i] for i in sample_indices])
    images_tensor = torch.stack(images)
    with torch.no_grad():
        outputs = model(images_tensor.to(device))
        probs = torch.nn.functional.softmax(outputs, dim=1)
        preds = (probs[:, 1] > 0.5).long().cpu().numpy()

    grid = make_grid(images_tensor, nrow=4)
    np_grid = grid.numpy().transpose((1, 2, 0))

    plt.figure(figsize=(14, 6))
    plt.imshow(np_grid)
    plt.axis("off")
    plt.title("Random Predictions on Test Set")
    for i in range(len(preds)):
        color = 'green' if preds[i] == labels[i] else 'red'
        plt.text((i % 4) * 60 + 20, (i // 4) * 60 + 10,
                 f"P:{preds[i]} A:{labels[i]}", color=color,
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--model_name", type=str, default="efficientnet_b2", help="EfficientNet variant")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_dir, args.model_name)
