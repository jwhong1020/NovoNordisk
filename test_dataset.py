#!/usr/bin/env python3
"""
Test script to verify dataset loading and check image formats.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from anomaly_datasets import NormalOnlyDataset
import matplotlib.pyplot as plt
import numpy as np

def test_dataset_loading():
    """Test if the dataset loads correctly and check image formats"""
    
    print("Testing dataset loading...")
    
    # Image transformations (same as in training)
    transforms_ = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    # Create dataset
    try:
        dataset = NormalOnlyDataset('OCT2017', transforms_=transforms_, mode='train')
        print(f"✅ Dataset created successfully with {len(dataset)} images")
        
        # Test first few samples
        print("Testing first 10 samples...")
        for i in range(min(10, len(dataset))):
            try:
                sample = dataset[i]
                image_A = sample['A']
                image_B = sample['B']
                path = sample['path']
                
                print(f"  Sample {i}: Shape A={image_A.shape}, Shape B={image_B.shape}, Path={path}")
                
                # Check if shapes are correct
                if image_A.shape != (3, 256, 256):
                    print(f"    ⚠️  Warning: Unexpected shape {image_A.shape}")
                else:
                    print(f"    ✅ Correct shape")
                    
            except Exception as e:
                print(f"    ❌ Error loading sample {i}: {e}")
                break
                
        # Test DataLoader
        print("\nTesting DataLoader...")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        
        try:
            batch = next(iter(dataloader))
            print(f"✅ DataLoader works! Batch shapes:")
            print(f"  A: {batch['A'].shape}")
            print(f"  B: {batch['B'].shape}")
            print(f"  Paths: {len(batch['path'])}")
            
            # Save a sample image to verify it looks correct
            sample_image = batch['A'][0]
            # Denormalize for display
            sample_image = (sample_image + 1.0) / 2.0  # From [-1,1] to [0,1]
            sample_image = torch.clamp(sample_image, 0, 1)
            
            # Convert to numpy and transpose for matplotlib
            sample_image = sample_image.permute(1, 2, 0).numpy()
            
            plt.figure(figsize=(8, 8))
            plt.imshow(sample_image)
            plt.title('Sample OCT Image')
            plt.axis('off')
            plt.savefig('sample_oct_image.png', dpi=150, bbox_inches='tight')
            print("✅ Sample image saved as 'sample_oct_image.png'")
            
        except Exception as e:
            print(f"❌ DataLoader error: {e}")
            
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return False
        
    print("\n🎉 Dataset testing completed successfully!")
    return True

if __name__ == '__main__':
    success = test_dataset_loading()
    if success:
        print("\n✅ Dataset is ready for training!")
        print("You can now run: python train_anomaly_simple.py")
    else:
        print("\n❌ Dataset has issues. Please check your data structure.")
