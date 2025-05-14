import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from efficientnet import get_efficientnet_model, load_model_from_checkpoint

def predict_single_image(image_path, model_path, model_name="efficientnet_b2"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model = get_efficientnet_model(model_name)
    model = load_model_from_checkpoint(model, model_path, device)
    model.eval()
    
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        prob = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, pred_class = torch.max(prob, 0)
    
    class_names = ["Normal", "Pneumonia"]
    print(f"Prediction: {class_names[pred_class]} (Confidence: {confidence.item() * 100:.2f}%)")
    return class_names[pred_class], confidence.item() * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to an X-ray image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model_name", type=str, default="efficientnet_b2", help="EfficientNet variant")
    args = parser.parse_args()
    
    predict_single_image(args.image_path, args.model_path, args.model_name)
