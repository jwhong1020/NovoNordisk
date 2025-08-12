#!/usr/bin/env python3
"""
Testing script for Abnormal-to-Normal Generator
Tests the model that "heals" abnormal images to look normal

Usage:
python test_abnormal_to_normal.py --dataset_name datasets/preprocessed_256 --model_epoch 29 --balance_dataset
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from tqdm import tqdm
import glob

from abnormal_to_normal_model import AbnormalToNormalDetector
from anomaly_datasets import AnomalyDatasetGrayscale
from utils import tensor2image

# Example usage:
# python test_abnormal_to_normal.py --dataset_name datasets/preprocessed_256 --model_epoch 29 --balance_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='datasets/preprocessed_256', help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--img_height', type=int, default=-1, help='size of image height')
    parser.add_argument('--img_width', type=int, default=-1, help='size of image width')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
    parser.add_argument('--model_epoch', type=int, default=29, help='epoch of trained model to load')
    parser.add_argument('--score_method', type=str, default='mse', choices=['mse', 'l1', 'combined'], help='anomaly scoring method')
    parser.add_argument('--threshold', type=float, default=None, help='anomaly threshold (if None, will be computed automatically)')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--border_margin', type=int, default=16, help='number of pixels to exclude from borders when calculating anomaly score')
    parser.add_argument('--use_concentration', type=bool, default=True, help='whether to apply concentration penalty for clustered anomalies')
    parser.add_argument('--concentration_weight', type=float, default=2.0, help='weight for concentration penalty')
    parser.add_argument('--concentration_method', type=str, default='centroid', choices=['centroid', 'patch'], help='method for computing concentration penalty')
    parser.add_argument('--balance_dataset', action='store_true', help='balance the test dataset by sampling equal numbers from normal and abnormal classes')
    parser.add_argument('--save_segmentation', action='store_true', help='save segmentation masks for visualization')
    
    opt = parser.parse_args()
    
    print("Abnormal-to-Normal Testing Configuration:")
    print(opt)

    # Create output directory with model epoch subfolder and clean previous results
    output_dir = f'output/abnormal_to_normal/{opt.dataset_name}/epoch_{opt.model_epoch}'
    os.makedirs(output_dir, exist_ok=True)

    # Clean previous images and results
    print("Cleaning previous test results...")
    try:
        # Remove previous image files
        for pattern in ['*.png', '*.csv']:
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
        print(f"  Error cleaning directory: {e}")

    print("Starting abnormal-to-normal anomaly detection evaluation...")

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Using device: {device}")

    # Initialize abnormal-to-normal detector
    model = AbnormalToNormalDetector(opt.channels, opt.channels, opt.n_residual_blocks)

    if cuda:
        model = model.cuda()

    # Load trained model
    model_path = f'saved_models/{opt.dataset_name}/abnormal_to_normal_{opt.model_epoch}.pth'

    if os.path.exists(model_path):
        # Load model weights (ignore segmentation head if present)
        checkpoint = torch.load(model_path, map_location=device)
        
        # Filter out segmentation head parameters if they exist
        filtered_checkpoint = {}
        for key, value in checkpoint.items():
            if not key.startswith('segmentation_head'):
                filtered_checkpoint[key] = value
        
        model.load_state_dict(filtered_checkpoint, strict=False)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model not found at {model_path}")
        exit(1)

    # Set to evaluation mode
    model.eval()

    # Image transformations
    transforms_ = []
    
    # Only resize if dimensions are specified
    if opt.img_height != -1 and opt.img_width != -1:
        transforms_.append(transforms.Resize((opt.img_height, opt.img_width)))
    
    transforms_.extend([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Test data loader
    dataset = AnomalyDatasetGrayscale(
        normal_dir=f"{opt.dataset_name}/test/NORMAL",
        abnormal_dir=f"{opt.dataset_name}/test/CNV",  # You can modify this to test other diseases
        transform=transforms.Compose(transforms_),
        balance_dataset=opt.balance_dataset
    )
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    def compute_optimal_threshold(scores, labels):
        """Compute optimal threshold using ROC curve"""
        fpr, tpr, thresholds = roc_curve(labels, scores)
        # Find threshold that maximizes (tpr - fpr)
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]

    def visualize_healing_results(images, healed_images, difference_maps, segmentation_masks, 
                                 scores, labels, paths, output_dir, save_top_n=5):
        """Visualize healing results with segmentation"""
        
        def mask2image(tensor):
            """Convert mask tensor (0-1 range) to image (0-255 range)"""
            mask = tensor[0].cpu().float().numpy()  # Get first channel
            mask = (mask * 255).astype(np.uint8)  # Convert 0-1 to 0-255
            if mask.ndim == 2:  # If 2D, make it 3D for consistency
                mask = np.expand_dims(mask, axis=0)
            return mask
        
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
        
        def save_healing_samples(indices, category_name, print_title):
            """Helper function to save a category of healing samples"""
            print(f"\n{print_title}:")
            for i in range(min(save_top_n, len(indices))):
                idx = indices[i]
                score = scores[idx]
                label = labels[idx]
                path = paths[idx]
                
                # Convert tensors to images
                original = tensor2image(images[idx])
                healed = tensor2image(healed_images[idx])
                diff = tensor2image(difference_maps[idx])
                seg_mask = mask2image(segmentation_masks[idx])  # Use special mask converter
                
                # Create visualization with 5 subplots: Original, Healed, Difference, Segmentation, Overlay
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                
                # Handle grayscale display
                if len(original.shape) == 3 and original.shape[0] == 1:
                    # Single channel grayscale
                    orig_gray = original[0]
                    healed_gray = healed[0]
                    diff_gray = diff[0]
                    seg_gray = seg_mask[0]
                elif len(original.shape) == 3 and original.shape[0] == 3:
                    # Multi-channel (take first channel)
                    orig_gray = original[0]
                    healed_gray = healed[0]
                    diff_gray = diff[0] 
                    seg_gray = seg_mask[0]
                else:
                    # Already 2D
                    orig_gray = original.squeeze()
                    healed_gray = healed.squeeze()
                    diff_gray = diff.squeeze()
                    seg_gray = seg_mask.squeeze()
                
                # Apply border cropping for visualization consistency
                if opt.border_margin > 0:
                    h, w = orig_gray.shape
                    if h > 2 * opt.border_margin and w > 2 * opt.border_margin:
                        orig_gray = orig_gray[opt.border_margin:-opt.border_margin, opt.border_margin:-opt.border_margin]
                        healed_gray = healed_gray[opt.border_margin:-opt.border_margin, opt.border_margin:-opt.border_margin]
                        diff_gray = diff_gray[opt.border_margin:-opt.border_margin, opt.border_margin:-opt.border_margin]
                        seg_gray = seg_gray[opt.border_margin:-opt.border_margin, opt.border_margin:-opt.border_margin]
                
                # Plot images
                axes[0].imshow(orig_gray, cmap='gray')
                axes[0].set_title(f'Original\n(Label: {"Anomaly" if label == 1 else "Normal"})')
                axes[0].axis('off')
                
                axes[1].imshow(healed_gray, cmap='gray')
                axes[1].set_title('Healed Image')
                axes[1].axis('off')
                
                axes[2].imshow(diff_gray, cmap='hot')
                axes[2].set_title(f'Healing Difference\n(Score: {score:.4f})')
                axes[2].axis('off')
                
                axes[3].imshow(seg_gray, cmap='hot', vmin=0, vmax=1)
                axes[3].set_title(f'Segmentation Mask\n(Max: {seg_gray.max():.4f}, Mean: {seg_gray.mean():.4f})')
                axes[3].axis('off')
                
                # Overlay segmentation on original - use simple threshold for binary mask
                overlay = orig_gray.copy()
                if seg_gray.max() > 0:
                    seg_threshold = 0.5  # Simple threshold since mask is already binary (0 or 1)
                else:
                    seg_threshold = 0.5
                    
                seg_binary = seg_gray > seg_threshold
                overlay = np.stack([overlay, overlay, overlay], axis=-1)  # Convert to RGB
                
                # Apply different colors based on label
                if label == 1:  # Anomaly
                    overlay[seg_binary, 0] = 1.0  # Red overlay for segmented regions
                else:  # Normal
                    overlay[seg_binary, 2] = 1.0  # Blue overlay (should be minimal)
                
                axes[4].imshow(overlay)
                axes[4].set_title(f'Segmentation Overlay\n(Threshold: {seg_threshold:.4f})')
                axes[4].axis('off')
                
                plt.tight_layout()
                filename = f'{category_name}_{i+1}_score_{score:.4f}_label_{label}.png'
                plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  {i+1}. Score: {score:.4f}, Label: {'Anomaly' if label == 1 else 'Normal'}, File: {os.path.basename(path)}")
        
        # Save top 5 anomalies (highest healing difference scores)
        save_healing_samples(sorted_anomaly_desc, "top_anomaly", f"Top {save_top_n} highest scoring anomalies")
        
        # Save top 5 normals (highest healing difference scores among normals)
        save_healing_samples(sorted_normal_desc, "top_normal", f"Top {save_top_n} highest scoring normals")
        
        # Save bottom 5 anomalies (lowest healing difference scores)
        save_healing_samples(sorted_anomaly_asc, "bottom_anomaly", f"Bottom {save_top_n} lowest scoring anomalies")

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
        print("ABNORMAL-TO-NORMAL ANOMALY DETECTION RESULTS")
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
        plt.title('ROC Curve for Abnormal-to-Normal Anomaly Detection')
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
        plt.xlabel('Healing Difference Score')
        plt.ylabel('Density')
        plt.title('Distribution of Healing Difference Scores')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/score_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return threshold, auc_score, report

    # ----------
    #  Testing
    # ----------

    print("Running abnormal-to-normal anomaly detection on test set...")

    all_scores = []
    all_labels = []
    all_images = []
    all_healed_images = []
    all_difference_maps = []
    all_segmentation_masks = []
    all_paths = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Get batch data
            images = Variable(batch['image']).to(device)
            labels = batch['label'].numpy()
            paths = batch['path']
            
            # Get model predictions
            results = model.detect_and_segment(
                images, 
                threshold=None,  # Don't threshold yet
                border_margin=opt.border_margin,
                method=opt.score_method,
                use_concentration=opt.use_concentration,
                concentration_weight=opt.concentration_weight,
                concentration_method=opt.concentration_method
            )
            
            # Store results
            all_scores.extend(results['anomaly_scores'].cpu().numpy())
            all_labels.extend(labels)
            all_images.extend(images.cpu())
            all_healed_images.extend(results['healed_images'].cpu())
            all_difference_maps.extend(results['difference_maps'].cpu())
            all_segmentation_masks.extend(results['segmentation_masks'].cpu())
            all_paths.extend(paths)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataloader)} batches")

    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    print(f"\nProcessed {len(all_scores)} images")
    print(f"Normal images: {np.sum(all_labels == 0)}")
    print(f"Anomalous images: {np.sum(all_labels == 1)}")

    # Evaluate performance
    threshold, auc_score, report = evaluate_performance(all_scores, all_labels, output_dir, opt.threshold)

    # Visualize healing results
    visualize_healing_results(all_images, all_healed_images, all_difference_maps, all_segmentation_masks,
                             all_scores, all_labels, all_paths, output_dir, save_top_n=5)

    # Save detailed results
    results_df = pd.DataFrame({
        'image_path': all_paths,
        'healing_difference_score': all_scores,
        'true_label': all_labels,
        'predicted_label': (all_scores > threshold).astype(int)
    })

    results_df.to_csv(f'{output_dir}/detailed_results.csv', index=False)

    print(f"\nResults saved to {output_dir}/")
    print("Files created:")
    print("  - detailed_results.csv: Detailed results for each image")
    print("  - roc_curve.png: ROC curve plot")
    print("  - score_distribution.png: Distribution of healing difference scores")
    print("  - top_anomaly_*.png: Top 5 highest scoring anomalies with healing visualization")
    print("  - top_normal_*.png: Top 5 highest scoring normals with healing visualization") 
    print("  - bottom_anomaly_*.png: Bottom 5 lowest scoring anomalies with healing visualization")
    if opt.save_segmentation:
        print("  - segmentation visualizations included in all sample images")
