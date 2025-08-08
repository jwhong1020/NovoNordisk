import glob
import random
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class AnomalyDataset(Dataset):
    """
    Dataset class for anomaly detection using CycleGAN.
    Loads only normal/healthy images for training, and normal + anomalous images for testing.
    """
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.mode = mode
        
        if mode == 'train':
            # For training, only load normal images
            self.files = sorted(glob.glob(os.path.join(root, 'train/NORMAL') + '/*.*'))
            print(f"Training with {len(self.files)} normal images")
        elif mode == 'test':
            # For testing, load all types of images (normal and anomalous)
            normal_files = sorted(glob.glob(os.path.join(root, 'test/NORMAL') + '/*.*'))
            cnv_files = sorted(glob.glob(os.path.join(root, 'test/CNV') + '/*.*'))
            dme_files = sorted(glob.glob(os.path.join(root, 'test/DME') + '/*.*'))
            drusen_files = sorted(glob.glob(os.path.join(root, 'test/DRUSEN') + '/*.*'))
            
            # Combine all files with labels
            self.files = []
            self.labels = []
            
            # Normal images (label 0)
            for file in normal_files:
                self.files.append(file)
                self.labels.append(0)  # 0 = normal
            
            # Anomalous images (label 1)
            for file in cnv_files + dme_files + drusen_files:
                self.files.append(file)
                self.labels.append(1)  # 1 = anomaly
                
            print(f"Testing with {len(normal_files)} normal and {len(cnv_files + dme_files + drusen_files)} anomalous images")

    def __getitem__(self, index):
        image_path = self.files[index % len(self.files)]
        
        # Open image and convert to RGB if needed
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image = self.transform(image)
        
        if self.mode == 'train':
            # For training, return the same image twice (A and B are the same)
            # This forces the model to learn identity mapping for normal images
            return {'A': image, 'B': image, 'path': image_path}
        else:
            # For testing, return image with its label
            label = self.labels[index % len(self.files)]
            return {'image': image, 'label': label, 'path': image_path}

    def __len__(self):
        return len(self.files)


class NormalOnlyDatasetGrayscale(Dataset):
    """
    Grayscale dataset that only loads normal images for self-supervised training.
    This is optimized for medical imaging where grayscale is the standard.
    """
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        
        # Load only normal images from the specified split
        if mode == 'train':
            self.files = sorted(glob.glob(os.path.join(root, 'train/NORMAL') + '/*.*'))
        elif mode == 'val':
            self.files = sorted(glob.glob(os.path.join(root, 'val/NORMAL') + '/*.*'))
        else:
            self.files = sorted(glob.glob(os.path.join(root, 'test/NORMAL') + '/*.*'))
            
        print(f"Loaded {len(self.files)} normal grayscale images for {mode}")

    def __getitem__(self, index):
        image_path = self.files[index % len(self.files)]
        
        # Open image and keep as grayscale (no RGB conversion)
        image = Image.open(image_path)
        if image.mode != 'L':  # Convert to grayscale if not already
            image = image.convert('L')
            
        image = self.transform(image)
        
        # Return the same image as both A and B for identity mapping
        return {'A': image, 'B': image, 'path': image_path}

    def __len__(self):
        return len(self.files)


class AnomalyDatasetGrayscale(Dataset):
    """
    Dataset class for anomaly detection using grayscale images.
    Contains both normal and abnormal images with labels.
    """
    
    def __init__(self, normal_dir, abnormal_dir, transform=None):
        self.transform = transform
        
        # Get all normal images
        normal_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            normal_files.extend(glob.glob(os.path.join(normal_dir, ext)))
        
        # Get all abnormal images  
        abnormal_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            abnormal_files.extend(glob.glob(os.path.join(abnormal_dir, ext)))
        
        # Combine files and create labels
        self.files = normal_files + abnormal_files
        self.labels = [0] * len(normal_files) + [1] * len(abnormal_files)  # 0=normal, 1=abnormal
        
        print(f"Loaded {len(normal_files)} normal and {len(abnormal_files)} abnormal grayscale images")
    
    def __getitem__(self, index):
        image_path = self.files[index]
        label = self.labels[index]
        
        # Load image and convert to grayscale
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'label': label, 'path': image_path}
    
    def __len__(self):
        return len(self.files)