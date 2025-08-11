#!/usr/bin/env python3

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import shutil
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

# Example usage:
# python test_anomaly.py --dataset_name datasets/preprocessed_resized --model_epoch 1 --model_type gray --border_margin 16 --balance_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='OCT2017', help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--img_height', type=int, default=-1, help='size of image height')
    parser.add_argument('--img_width', type=int, default=-1, help='size of image width')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
    parser.add_argument('--model_epoch', type=int, default=200, help='epoch of trained model to load')
    parser.add_argument('--score_method', type=str, default='mse', choices=['mse', 'l1', 'combined'], help='anomaly scoring method')
    parser.add_argument('--threshold', type=float, default=None, help='anomaly threshold (if None, will be computed automatically)')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--model_type', type=str, default='gray', choices=['rgb', 'gray'], help='model type (rgb or grayscale)')
    parser.add_argument('--border_margin', type=int, default=16, help='number of pixels to exclude from borders when calculating anomaly score')
    parser.add_argument('--use_concentration', type=bool, default=True, help='whether to apply concentration penalty for clustered anomalies')
    parser.add_argument('--concentration_weight', type=float, default=2.0, help='weight for concentration penalty (higher = more penalty for concentration)')
    parser.add_argument('--concentration_method', type=str, default='centroid', choices=['centroid', 'patch'], help='method for computing concentration penalty')
    parser.add_argument('--balance_dataset', action='store_true', help='balance the test dataset by sampling equal numbers from normal and abnormal classes')
    opt = parser.parse_args()
    print(opt)

    # Adjust channels for grayscale
    if opt.model_type == 'gray':
        opt.channels = 1
        print(f"Using grayscale model with {opt.channels} channel(s)")
    else:
        print(f"Using RGB model with {opt.channels} channel(s)")

# Create output directory with model epoch subfolder and clean previous results
output_dir = f'output/anomaly_detection/{opt.dataset_name}/epoch_{opt.model_epoch}'
os.makedirs(output_dir, exist_ok=True)

# Clean previous images and results
print("Cleaning previous test results...")
try:
    # Remove previous image files
    for pattern in ['top_anomaly_*.png', 'roc_curve.png', 'score_distribution.png', 'detailed_results.csv']:
        files_to_remove = glob.glob(os.path.join(output_dir, pattern))
        for file_path in files_to_remove:
            os.remove(file_path)
            print(f"  Removed: {os.path.basename(file_path)}")
    
    if len(glob.glob(os.path.join(output_dir, '*'))) == 0:
        print("  Output directory is now clean")
    else:
        remaining_files = glob.glob(os.path.join(output_dir, '*'))
        print(f"  {len(remaining_files)} other files remain in output directory")
        
except Exception as e:
    print(f"  Warning: Could not clean some files: {e}")

print("Starting anomaly detection evaluation...")

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

# Modified transforms for original size preservation
transforms_ = []

# Only resize if dimensions are specified and different from a "preserve" flag
if opt.img_height != -1 and opt.img_width != -1:  # Could add -1 as "preserve original"
    transforms_.append(transforms.Resize((opt.img_height, opt.img_width), transforms.InterpolationMode.BICUBIC))


# Image transformations
if opt.model_type == 'gray':
    transforms_.extend([ transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)) ])
else:
    transforms_.extend([ transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ])

# Test data loader
if opt.model_type == 'gray':
    from anomaly_datasets import AnomalyDatasetGrayscale
    # For grayscale testing, use the specific dataset class
    dataset = AnomalyDatasetGrayscale(
        normal_dir=f"{opt.dataset_name}/test/NORMAL",
        abnormal_dir=f"{opt.dataset_name}/test/CNV", 
        transform=transforms.Compose(transforms_),
        balance_dataset=opt.balance_dataset
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

def visualize_results(images, reconstructed, difference_maps, scores, labels, paths, output_dir, border_margin=16, save_top_n=5):
    """Visualize top 5 anomalies, top 5 normals, and bottom 5 anomalies with border cropping"""
    
    # Convert to numpy arrays for easier indexing
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Separate anomaly and normal indices
    anomaly_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]
    
    # Sort anomaly scores (descending for top, ascending for bottom)
    anomaly_scores = scores[anomaly_indices]
    normal_scores = scores[normal_indices]
    
    sorted_anomaly_desc = anomaly_indices[np.argsort(anomaly_scores)[::-1]]  # Top anomalies
    sorted_anomaly_asc = anomaly_indices[np.argsort(anomaly_scores)]         # Bottom anomalies
    sorted_normal_desc = normal_indices[np.argsort(normal_scores)[::-1]]     # Top normals
    
    def save_samples(indices, category_name, print_title):
        """Helper function to save a category of samples"""
        print(f"\n{print_title}:")
        for i in range(min(save_top_n, len(indices))):
            idx = indices[i]
            score = scores[idx]
            label = labels[idx]
            path = paths[idx]
            
            # Convert tensors to images
            original = tensor2image(images[idx])
            recon = tensor2image(reconstructed[idx])
            diff = tensor2image(difference_maps[idx])
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Handle grayscale vs RGB display
            if opt.model_type == 'gray':
                # For grayscale, the tensor2image function tiles to 3 channels
                if len(original.shape) == 3 and original.shape[0] == 3:
                    orig_gray = original[0]  # Shape: (H, W)
                    recon_gray = recon[0]    # Shape: (H, W)  
                    diff_gray = diff[0]      # Shape: (H, W)
                else:
                    orig_gray = original.squeeze() if len(original.shape) > 2 else original
                    recon_gray = recon.squeeze() if len(recon.shape) > 2 else recon
                    diff_gray = diff.squeeze() if len(diff.shape) > 2 else diff
                
                # Crop border regions to match the anomaly score calculation
                if border_margin > 0:
                    h, w = orig_gray.shape
                    if h > 2 * border_margin and w > 2 * border_margin:
                        orig_gray = orig_gray[border_margin:-border_margin, border_margin:-border_margin]
                        recon_gray = recon_gray[border_margin:-border_margin, border_margin:-border_margin]
                        diff_gray = diff_gray[border_margin:-border_margin, border_margin:-border_margin]
                        
                axes[0].imshow(orig_gray, cmap='gray')
                axes[1].imshow(recon_gray, cmap='gray')
                axes[2].imshow(diff_gray, cmap='hot')
            else:
                # For RGB, transpose and display normally
                orig_rgb = np.transpose(original, (1, 2, 0))
                recon_rgb = np.transpose(recon, (1, 2, 0))
                diff_rgb = np.transpose(diff, (1, 2, 0))
                
                # Crop border regions to match the anomaly score calculation
                if border_margin > 0:
                    h, w = orig_rgb.shape[:2]
                    if h > 2 * border_margin and w > 2 * border_margin:
                        orig_rgb = orig_rgb[border_margin:-border_margin, border_margin:-border_margin]
                        recon_rgb = recon_rgb[border_margin:-border_margin, border_margin:-border_margin]
                        diff_rgb = diff_rgb[border_margin:-border_margin, border_margin:-border_margin]
                
                axes[0].imshow(orig_rgb)
                axes[1].imshow(recon_rgb)
                axes[2].imshow(diff_rgb)
                
            axes[0].set_title(f'Original (Label: {"Anomaly" if label == 1 else "Normal"})\nCentral Region Only')
            axes[0].axis('off')
            
            axes[1].set_title('Reconstructed\nCentral Region Only')
            axes[1].axis('off')
            
            axes[2].set_title(f'Difference (Score: {score:.4f})\nCentral Region Only')
            axes[2].axis('off')
            
            plt.tight_layout()
            filename = f'{category_name}_{i+1}_score_{score:.4f}_label_{label}.png'
            plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  {i+1}. Score: {score:.4f}, Label: {'Anomaly' if label == 1 else 'Normal'}, File: {os.path.basename(path)}")
    
    # Save top 5 anomalies (highest scores among anomalies)
    save_samples(sorted_anomaly_desc, "top_anomaly", f"Top {save_top_n} highest scoring anomalies")
    
    # Save top 5 normals (highest scores among normals)
    save_samples(sorted_normal_desc, "top_normal", f"Top {save_top_n} highest scoring normals")
    
    # Save bottom 5 anomalies (lowest scores among anomalies)
    save_samples(sorted_anomaly_asc, "bottom_anomaly", f"Bottom {save_top_n} lowest scoring anomalies")

def evaluate_performance(scores, labels, output_dir, threshold=None):
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
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=150, bbox_inches='tight')
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
    plt.savefig(f'{output_dir}/score_distribution.png', dpi=150, bbox_inches='tight')
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
        
        # Detect anomalies with concentration penalty
        results = anomaly_detector.detect_anomalies(
            real_images, 
            border_margin=opt.border_margin,
            method=opt.score_method,
            use_concentration=opt.use_concentration,
            concentration_weight=opt.concentration_weight,
            concentration_method=opt.concentration_method
        )
        
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
threshold, auc_score, report = evaluate_performance(all_scores, all_labels, output_dir, opt.threshold)

# Visualize results with border cropping
visualize_results(all_images, all_reconstructed, all_difference_maps, 
                 all_scores, all_labels, all_paths, output_dir, border_margin=opt.border_margin, save_top_n=5)

# Save detailed results
results_df = pd.DataFrame({
    'image_path': all_paths,
    'anomaly_score': all_scores,
    'true_label': all_labels,
    'predicted_label': (all_scores > threshold).astype(int)
})

results_df.to_csv(f'{output_dir}/detailed_results.csv', index=False)

print(f"\nResults saved to {output_dir}/")
print("Files created:")
print("  - detailed_results.csv: Detailed results for each image")
print("  - roc_curve.png: ROC curve plot")
print("  - score_distribution.png: Distribution of anomaly scores")
print("  - top_anomaly_*.png: Top 5 highest scoring anomalies")
print("  - top_normal_*.png: Top 5 highest scoring normals") 
print("  - bottom_anomaly_*.png: Bottom 5 lowest scoring anomalies")
