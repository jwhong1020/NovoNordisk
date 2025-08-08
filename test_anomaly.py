#!/usr/bin/env python3

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, classification_report
import pandas as pd

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from anomaly_models import AnomalyDetector
from anomaly_datasets import AnomalyDataset
from utils import tensor2image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='OCT2017', help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--img_height', type=int, default=256, help='size of image height')
    parser.add_argument('--img_width', type=int, default=256, help='size of image width')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
    parser.add_argument('--model_epoch', type=int, default=200, help='epoch of trained model to load')
    parser.add_argument('--score_method', type=str, default='mse', choices=['mse', 'l1', 'combined'], help='anomaly scoring method')
    parser.add_argument('--threshold', type=float, default=None, help='anomaly threshold (if None, will be computed automatically)')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--model_type', type=str, default='rgb', choices=['rgb', 'gray'], help='model type (rgb or grayscale)')
    opt = parser.parse_args()
    print(opt)

    # Adjust channels for grayscale
    if opt.model_type == 'gray':
        opt.channels = 1
        print(f"Using grayscale model with {opt.channels} channel(s)")
    else:
        print(f"Using RGB model with {opt.channels} channel(s)")

# Create output directory
os.makedirs('output/anomaly_detection/%s' % opt.dataset_name, exist_ok=True)

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Initialize anomaly detector
anomaly_detector = AnomalyDetector(opt.channels, opt.channels, opt.n_residual_blocks)

if cuda:
    anomaly_detector = anomaly_detector.cuda()

# Load trained model
if opt.model_type == 'gray':
    model_path = 'saved_models/%s/anomaly_detector_gray_%d.pth' % (opt.dataset_name, opt.model_epoch)
else:
    model_path = 'saved_models/%s/anomaly_detector_%d.pth' % (opt.dataset_name, opt.model_epoch)

if os.path.exists(model_path):
    anomaly_detector.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")
else:
    print(f"Model not found at {model_path}")
    exit(1)

# Set to evaluation mode
anomaly_detector.eval()

# Image transformations
if opt.model_type == 'gray':
    transforms_ = [ transforms.Resize((opt.img_height, opt.img_width), transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)) ]
else:
    transforms_ = [ transforms.Resize((opt.img_height, opt.img_width), transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

# Test data loader
if opt.model_type == 'gray':
    from anomaly_datasets import AnomalyDatasetGrayscale
    # For grayscale testing, use the specific dataset class
    dataset = AnomalyDatasetGrayscale(
        normal_dir=f"{opt.dataset_name}/test/NORMAL",
        abnormal_dir=f"{opt.dataset_name}/test/CNV", 
        transform=transforms.Compose(transforms_)
    )
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
else:
    dataloader = DataLoader(AnomalyDataset(opt.dataset_name, transforms_=transforms_, mode='test'),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

def compute_optimal_threshold(scores, labels):
    """Compute optimal threshold using ROC curve"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Find threshold that maximizes (tpr - fpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def visualize_results(images, reconstructed, difference_maps, scores, labels, paths, save_top_n=10):
    """Visualize top anomalies and normal samples"""
    
    # Sort by anomaly score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    
    # Save top anomalies
    print(f"\nTop {save_top_n} highest scoring samples:")
    for i in range(min(save_top_n, len(sorted_indices))):
        idx = sorted_indices[i]
        score = scores[idx]
        label = labels[idx]
        path = paths[idx]
        
        # Convert tensors to images
        original = tensor2image(images[idx])
        recon = tensor2image(reconstructed[idx])
        diff = tensor2image(difference_maps[idx])
        
        # Debug shapes for grayscale
        if opt.model_type == 'gray':
            print(f"Debug shapes - Original: {original.shape}, Recon: {recon.shape}, Diff: {diff.shape}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Handle grayscale vs RGB display
        if opt.model_type == 'gray':
            # For grayscale, the tensor2image function tiles to 3 channels
            # So we need to use the original format and handle it properly
            if len(original.shape) == 3 and original.shape[0] == 3:
                # tensor2image tiled the grayscale to RGB, so take one channel
                orig_gray = original[0]  # Shape: (H, W)
                recon_gray = recon[0]    # Shape: (H, W)  
                diff_gray = diff[0]      # Shape: (H, W)
            else:
                # Handle other cases
                orig_gray = original.squeeze() if len(original.shape) > 2 else original
                recon_gray = recon.squeeze() if len(recon.shape) > 2 else recon
                diff_gray = diff.squeeze() if len(diff.shape) > 2 else diff
                
            axes[0].imshow(orig_gray, cmap='gray')
            axes[1].imshow(recon_gray, cmap='gray')
            axes[2].imshow(diff_gray, cmap='hot')  # Use hot colormap for difference
        else:
            # For RGB, transpose and display normally
            axes[0].imshow(np.transpose(original, (1, 2, 0)))
            axes[1].imshow(np.transpose(recon, (1, 2, 0)))
            axes[2].imshow(np.transpose(diff, (1, 2, 0)))
            
        axes[0].set_title(f'Original (Label: {"Anomaly" if label == 1 else "Normal"})')
        axes[0].axis('off')
        
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        
        axes[2].set_title(f'Difference (Score: {score:.4f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        filename = f'top_anomaly_{i+1}_score_{score:.4f}_label_{label}.png'
        plt.savefig(f'output/anomaly_detection/{opt.dataset_name}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  {i+1}. Score: {score:.4f}, Label: {'Anomaly' if label == 1 else 'Normal'}, File: {os.path.basename(path)}")

def evaluate_performance(scores, labels, threshold=None):
    """Evaluate anomaly detection performance"""
    
    if threshold is None:
        threshold = compute_optimal_threshold(scores, labels)
    
    predictions = (scores > threshold).astype(int)
    
    # Compute metrics
    auc_score = roc_auc_score(labels, scores)
    
    # Classification report
    report = classification_report(labels, predictions, 
                                 target_names=['Normal', 'Anomaly'], 
                                 output_dict=True)
    
    print(f"\n{'='*50}")
    print("ANOMALY DETECTION RESULTS")
    print(f"{'='*50}")
    print(f"Optimal Threshold: {threshold:.4f}")
    print(f"AUC-ROC Score: {auc_score:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['Normal', 'Anomaly']))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Anomaly Detection')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'output/anomaly_detection/{opt.dataset_name}/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot score distribution
    plt.figure(figsize=(10, 6))
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', density=True)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'output/anomaly_detection/{opt.dataset_name}/score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return threshold, auc_score, report

# ----------
#  Testing
# ----------

print("Running anomaly detection on test set...")

all_scores = []
all_labels = []
all_images = []
all_reconstructed = []
all_difference_maps = []
all_paths = []

with torch.no_grad():
    for i, batch in enumerate(dataloader):
        # Get test data
        real_images = Variable(batch['image'].type(Tensor))
        labels = batch['label'].numpy()
        paths = batch['path']
        
        # Detect anomalies
        results = anomaly_detector.detect_anomalies(real_images)
        
        # Store results
        scores = results['anomaly_scores'].cpu().numpy()
        reconstructed = results['reconstructed']
        difference_maps = results['difference_maps']
        
        all_scores.extend(scores)
        all_labels.extend(labels)
        all_images.extend(real_images.cpu())
        all_reconstructed.extend(reconstructed.cpu())
        all_difference_maps.extend(difference_maps.cpu())
        all_paths.extend(paths)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(dataloader)} batches")

# Convert to numpy arrays
all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

print(f"\nProcessed {len(all_scores)} images")
print(f"Normal images: {np.sum(all_labels == 0)}")
print(f"Anomalous images: {np.sum(all_labels == 1)}")

# Evaluate performance
threshold, auc_score, report = evaluate_performance(all_scores, all_labels, opt.threshold)

# Visualize results
visualize_results(all_images, all_reconstructed, all_difference_maps, 
                 all_scores, all_labels, all_paths, save_top_n=15)

# Save detailed results
results_df = pd.DataFrame({
    'image_path': all_paths,
    'anomaly_score': all_scores,
    'true_label': all_labels,
    'predicted_label': (all_scores > threshold).astype(int)
})

results_df.to_csv(f'output/anomaly_detection/{opt.dataset_name}/detailed_results.csv', index=False)

print(f"\nResults saved to output/anomaly_detection/{opt.dataset_name}/")
print("Files created:")
print("  - detailed_results.csv: Detailed results for each image")
print("  - roc_curve.png: ROC curve plot")
print("  - score_distribution.png: Distribution of anomaly scores")
print("  - top_anomaly_*.png: Visualizations of highest scoring samples")
