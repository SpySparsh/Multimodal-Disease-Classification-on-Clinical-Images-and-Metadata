import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2
import numpy as np

class CXRMultimodalDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        """
        Args:
            csv_file (str): Path to metadata CSV.
            images_dir (str): Path to folder with X-ray images.
            transform (callable, optional): Optional torchvision transforms for images.
        """
        self.metadata = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform

        # Encode labels as integers
        self.classes = self.metadata['finding'].unique().tolist()
        self.class2idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['finding'].map(self.class2idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load image
        img_path = os.path.join(self.images_dir, row['filename'])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # X-ray is grayscale
        image = cv2.resize(image, (224, 224))  # Resize for CNN
        image = image.astype(np.float32) / 255.0  # Normalize to [0,1]

        # Add channel dimension (C, H, W)
        image = np.expand_dims(image, axis=0)
        if self.transform:
            image = self.transform(image)

        image_tensor = torch.tensor(image, dtype=torch.float)

        # Process metadata (age, sex)
        # Convert 'M'/'F' to 0/1
        sex = 0 if row['sex'] == 'M' else 1
        age = float(row['age']) if not pd.isna(row['age']) else 0.0
        metadata_features = np.array([age, sex], dtype=np.float32)
        metadata_tensor = torch.tensor(metadata_features, dtype=torch.float)

        # Label
        label = torch.tensor(row['label'], dtype=torch.long)

        return {
            'image': image_tensor,
            'metadata': metadata_tensor,
            'label': label
        }
