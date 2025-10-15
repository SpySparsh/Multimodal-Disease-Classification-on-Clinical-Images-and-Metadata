import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=128, pretrained=True):
        super(ImageEncoder, self).__init__()
        
        # Load pretrained ResNet18
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # until avgpool

        # Freeze early layers (optional for transfer learning)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Add a projection layer to map features into smaller embedding
        in_features = base_model.fc.in_features  # usually 512 for ResNet-18
        self.fc = nn.Linear(in_features, output_dim)

    def forward(self, x):
        x = self.feature_extractor(x)     # shape: [B, 512, 1, 1]
        x = torch.flatten(x, 1)           # shape: [B, 512]
        x = self.fc(x)                    # shape: [B, output_dim]
        return x
