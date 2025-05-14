import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from data_loader import get_dataloaders
from efficientnet import get_efficientnet_model

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.0005, weight_decay=1e-5,
                model_dir="models", model_name="efficientnet_b2_pneumonia"):
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Compute class weights correctly
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    
    all_labels = np.array(all_labels)
    class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels), dtype=torch.float).to(device)
    
    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=num_epochs, pct_start=0.1)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 7
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total * 100
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, f"{model_name}_best.pth"))
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered")
                break
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/total:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    return model

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return val_loss / total, correct / total * 100

if __name__ == "__main__":
    data_dir = "../data/chest_xray"
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=32)
    model = get_efficientnet_model("efficientnet_b2")
    train_model(model, train_loader, val_loader)