import torch
import torch.nn as nn
import torchvision.models as models

def get_efficientnet_model(model_name="efficientnet_b2", num_classes=2, pretrained=True):
    available_models = [
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
        "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"
    ]
    
    if model_name not in available_models:
        print(f"Warning: {model_name} not available. Defaulting to efficientnet_b2.")
        model_name = "efficientnet_b2"
    
    model_fn = getattr(models, model_name)
    model = model_fn(pretrained=pretrained)
    
    if hasattr(model, "classifier"):
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(512, num_classes)
        )

    else:
        raise AttributeError("Unexpected model architecture - no classifier found.")
    
    return model

def load_model_from_checkpoint(model, checkpoint_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get('state_dict', checkpoint))
        print(f"Model loaded from {checkpoint_path}")
        return model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return model.to(device)