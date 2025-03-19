import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

def precompute_features(model: nn.Module, dataset: torch.utils.data.Dataset, device: torch.device) -> torch.utils.data.Dataset:
    """
    Precompute features by replacing the last layer with an identity.
    """
    original_fc = model.fc
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    
    all_features = []
    all_labels = []
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            features = model(x)
            all_features.append(features.cpu())
            all_labels.append(y)
    
    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    
    model.fc = original_fc
    return TensorDataset(all_features, all_labels)

class LastLayer(nn.Module):
    def __init__(self):
        super(LastLayer, self).__init__()
        self.fc = nn.Linear(512, 2)  # For ResNet18
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()
        self.resnet = models.resnet18(weights="DEFAULT")
        self.resnet.fc = LastLayer()
        # You can add further modifications here if desired.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
