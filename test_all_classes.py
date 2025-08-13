#!/usr/bin/env python3
"""
Comprehensive testing script for Abnormal-to-Normal Generator on all disease classes
Tests the model on CNV, DME, DRUSEN, and NORMAL images

Usage:
python test_all_classes.py --dataset_name datasets/preprocessed_128 --model_epoch 185 --balance_dataset
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
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from tqdm import tqdm
import glob
import seaborn as sns

from abnormal_to_normal_model import AbnormalToNormalDetector
from anomaly_datasets import AnomalyDatasetGrayscale
from utils import tensor2image

# Disease classes to test
DISEASE_CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def create_multi_class_dataset(dataset_name, transforms_, balance_dataset=False):
    """Create dataset with all disease classes"""
    all_data = []
    class_counts = {}
    
    # First pass: collect all data and count samples
    all_class_data = {}
    for class_idx, class_name in enumerate(DISEASE_CLASSES):
        class_dir = f"{dataset_name}/test/{class_name}"
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found, skipping {class_name}")
            continue
            
        # Create dataset for this class
        if class_name == 'NORMAL':
            # Normal class (label = 0)
            dataset = AnomalyDatasetGrayscale(
                normal_dir=class_dir,
                abnormal_dir=None,  # Only normal
                transform=transforms.Compose(transforms_),
                mode='normal_only'
            )
            label = 0
        else:
            # Abnormal classes (label = 1, but we'll track specific disease)
            dataset = AnomalyDatasetGrayscale(
                normal_dir=None,
                abnormal_dir=class_dir,
                transform=transforms.Compose(transforms_),
                mode='abnormal_only'
            )
            label = 1
        
        # Collect all samples from this class
        class_samples = []
        for i in range(len(dataset)):
            sample = dataset[i]
            sample['disease_class'] = class_name
            sample['disease_idx'] = class_idx
            sample['binary_label'] = label  # 0 for normal, 1 for abnormal
            class_samples.append(sample)
        
        all_class_data[class_name] = class_samples
        class_counts[class_name] = len(dataset)
        print(f"Loaded {len(dataset)} images from {class_name}")
    
    # Balance dataset if requested
    if balance_dataset and len(class_counts) > 0:
        # Custom balancing: each disease class gets 1/3 of normal images
        normal_count = class_counts.get('NORMAL', 0)
        disease_classes = ['CNV', 'DME', 'DRUSEN']
        available_diseases = [cls for cls in disease_classes if cls in class_counts]
        
        if normal_count > 0 and len(available_diseases) > 0:
            # Each disease class gets 1/3 of normal samples
            samples_per_disease = normal_count // 3
            print(f"\nCustom balancing: {normal_count} normal images, {samples_per_disease} per disease class")
            
            balanced_data = []
            
            # Add all normal samples
            if 'NORMAL' in all_class_data:
                balanced_data.extend(all_class_data['NORMAL'])
                print(f"  NORMAL: {len(all_class_data['NORMAL'])} samples")
            
            # Add balanced disease samples
            for disease_class in available_diseases:
                if disease_class in all_class_data:
                    disease_samples = all_class_data[disease_class]
                    if len(disease_samples) >= samples_per_disease:
                        # Randomly sample the required number
                        indices = np.random.choice(len(disease_samples), samples_per_disease, replace=False)
                        selected_samples = [disease_samples[i] for i in indices]
                    else:
                        # Use all available samples if not enough
                        selected_samples = disease_samples
                        print(f"  Warning: {disease_class} has only {len(disease_samples)} samples, using all")
                    
                    balanced_data.extend(selected_samples)
                    print(f"  {disease_class}: {len(selected_samples)} samples")
            
            all_data = balanced_data
            
            # Update class counts for balanced dataset
            balanced_counts = {}
            for sample in balanced_data:
                cls = sample['disease_class']
                balanced_counts[cls] = balanced_counts.get(cls, 0) + 1
            
            print(f"\nBalanced dataset summary:")
            print(f"  Total samples: {len(all_data)}")
            for cls, count in balanced_counts.items():
                print(f"  {cls}: {count} samples")
                
            return all_data, balanced_counts
        else:
            print("Warning: Cannot balance dataset - missing normal or disease classes")
    
    # If not balancing, add all data
    for class_samples in all_class_data.values():
        all_data.extend(class_samples)
    
    return all_data, class_counts

class MultiClassDataset(torch.utils.data.Dataset):
    """Custom dataset wrapper for multi-class data"""
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def visualize_multi_class_results(images, healed_images, difference_maps, segmentation_masks, 
                                 scores, labels, disease_classes, paths, output_dir, save_per_class=3):
    """Visualize healing results for all disease classes"""
    
    def mask2image(tensor):
        """Convert mask tensor to image"""
        mask = tensor[0].cpu().float().numpy()
        mask = (mask * 255).astype(np.uint8)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
        return mask
    
    # Convert to numpy arrays
    scores = np.array(scores)
    labels = np.array(labels)
    disease_classes = np.array(disease_classes)
    
    # Process each disease class
    for class_name in DISEASE_CLASSES:
        class_mask = disease_classes == class_name
        if not np.any(class_mask):
            continue
            
        class_indices = np.where(class_mask)[0]
        class_scores = scores[class_mask]
        
        # Sort by score (descending for top examples)
        sorted_indices = class_indices[np.argsort(class_scores)[::-1]]
        
        print(f"\nTop {save_per_class} examples for {class_name}:")
        
        for i in range(min(save_per_class, len(sorted_indices))):
            idx = sorted_indices[i]
            score = scores[idx]
            label = labels[idx]
            path = paths[idx]
            
            # Convert tensors to images
            original = tensor2image(images[idx])
            healed = tensor2image(healed_images[idx])
            diff = tensor2image(difference_maps[idx])
            seg_mask = mask2image(segmentation_masks[idx])
            
            # Create visualization
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            # Handle grayscale display
            if len(original.shape) == 3 and original.shape[0] == 1:
                orig_gray = original[0]
                healed_gray = healed[0]
                diff_gray = diff[0]
                seg_gray = seg_mask[0]
            else:
                orig_gray = original.squeeze()
                healed_gray = healed.squeeze()
                diff_gray = diff.squeeze()
                seg_gray = seg_mask.squeeze()
            
            # Plot images
            axes[0].imshow(orig_gray, cmap='gray')
            axes[0].set_title(f'Original {class_name}\n(Label: {"Abnormal" if label == 1 else "Normal"})')
            axes[0].axis('off')
            
            axes[1].imshow(healed_gray, cmap='gray')
            axes[1].set_title('Healed Image')
            axes[1].axis('off')
            
            axes[2].imshow(diff_gray, cmap='hot')
            axes[2].set_title(f'Healing Difference\n(Score: {score:.4f})')
            axes[2].axis('off')
            
            axes[3].imshow(seg_gray, cmap='hot', vmin=0, vmax=1)
            axes[3].set_title(f'Segmentation Mask\n(Max: {seg_gray.max():.4f})')
            axes[3].axis('off')
            
            # Overlay
            overlay = np.stack([orig_gray, orig_gray, orig_gray], axis=-1)
            seg_binary = seg_gray > 0.5
            if label == 1:  # Abnormal
                overlay[seg_binary, 0] = 1.0  # Red overlay
            else:  # Normal
                overlay[seg_binary, 2] = 1.0  # Blue overlay
            
            axes[4].imshow(overlay)
            axes[4].set_title('Segmentation Overlay')
            axes[4].axis('off')
            
            plt.tight_layout()
            filename = f'{class_name}_example_{i+1}_score_{score:.4f}.png'
            plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  {i+1}. Score: {score:.4f}, File: {os.path.basename(path)}")

def evaluate_multi_class_performance(scores, binary_labels, disease_classes, output_dir):
    """Evaluate performance across all classes"""
    
    # Binary classification (Normal vs All Abnormal)
    threshold = compute_optimal_threshold(scores, binary_labels)
    predictions = (scores > threshold).astype(int)
    
    # Overall metrics
    auc_score = roc_auc_score(binary_labels, scores)
    
    print(f"\n{'='*60}")
    print("MULTI-CLASS ABNORMAL-TO-NORMAL DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Optimal Threshold: {threshold:.4f}")
    print(f"Overall AUC-ROC Score: {auc_score:.4f}")
    
    # Binary classification report
    print(f"\nBinary Classification Report (Normal vs Abnormal):")
    print(classification_report(binary_labels, predictions, target_names=['Normal', 'Abnormal']))
    
    # Per-class analysis
    print(f"\nPer-Class Analysis:")
    print("-" * 40)
    
    class_results = {}
    for class_name in DISEASE_CLASSES:
        class_mask = disease_classes == class_name
        if not np.any(class_mask):
            continue
            
        class_scores = scores[class_mask]
        class_binary_labels = binary_labels[class_mask]
        
        mean_score = np.mean(class_scores)
        std_score = np.std(class_scores)
        median_score = np.median(class_scores)
        
        if class_name == 'NORMAL':
            # For normal class, we want LOW scores (good healing = low difference)
            detection_rate = np.mean(class_scores <= threshold)
            print(f"\n{class_name}:")
            print(f"  Mean Score: {mean_score:.4f} ± {std_score:.4f}")
            print(f"  Median Score: {median_score:.4f}")
            print(f"  Correctly Classified (≤ threshold): {detection_rate:.2%}")
        else:
            # For abnormal classes, we want HIGH scores (good healing = high difference)
            detection_rate = np.mean(class_scores > threshold)
            print(f"\n{class_name}:")
            print(f"  Mean Score: {mean_score:.4f} ± {std_score:.4f}")
            print(f"  Median Score: {median_score:.4f}")
            print(f"  Detection Rate (> threshold): {detection_rate:.2%}")
        
        class_results[class_name] = {
            'mean_score': mean_score,
            'std_score': std_score,
            'median_score': median_score,
            'detection_rate': detection_rate,
            'count': np.sum(class_mask)
        }
    
    # Create visualizations
    create_multi_class_visualizations(scores, binary_labels, disease_classes, threshold, output_dir, class_results)
    
    return threshold, auc_score, class_results

def create_multi_class_visualizations(scores, binary_labels, disease_classes, threshold, output_dir, class_results):
    """Create comprehensive visualizations for multi-class results"""
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(binary_labels, scores)
    auc_score = roc_auc_score(binary_labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Multi-Class Abnormal-to-Normal Detection')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/multi_class_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Score distributions by class
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange']
    for i, class_name in enumerate(DISEASE_CLASSES):
        class_mask = disease_classes == class_name
        if not np.any(class_mask):
            continue
        class_scores = scores[class_mask]
        plt.hist(class_scores, bins=30, alpha=0.7, label=class_name, color=colors[i % len(colors)])
    
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Healing Difference Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution by Disease Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/multi_class_score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Box plot by class
    plt.figure(figsize=(10, 6))
    
    class_score_lists = []
    class_names = []
    for class_name in DISEASE_CLASSES:
        class_mask = disease_classes == class_name
        if np.any(class_mask):
            class_score_lists.append(scores[class_mask])
            class_names.append(class_name)
    
    plt.boxplot(class_score_lists, labels=class_names)
    plt.axhline(threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold:.4f})')
    plt.ylabel('Healing Difference Score')
    plt.title('Score Distribution by Disease Class (Box Plot)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/multi_class_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Performance summary bar chart
    plt.figure(figsize=(10, 6))
    
    classes = list(class_results.keys())
    detection_rates = [class_results[c]['detection_rate'] for c in classes]
    
    bars = plt.bar(classes, detection_rates, color=['blue' if c == 'NORMAL' else 'red' for c in classes], alpha=0.7)
    plt.ylabel('Detection/Classification Rate')
    plt.title('Performance by Disease Class')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, detection_rates):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{rate:.2%}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/multi_class_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Confusion matrix (simplified: Normal vs Abnormal)
    predictions = (scores > threshold).astype(int)
    cm = confusion_matrix(binary_labels, predictions)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'], 
                yticklabels=['Normal', 'Abnormal'])
    plt.title('Confusion Matrix (Normal vs Abnormal)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

def compute_optimal_threshold(scores, labels):
    """Compute optimal threshold using ROC curve"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

def main():
    parser = argparse.ArgumentParser(description='Multi-class testing for Abnormal-to-Normal Generator')
    parser.add_argument('--dataset_name', type=str, default='datasets/preprocessed_128', help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--img_height', type=int, default=-1, help='size of image height')
    parser.add_argument('--img_width', type=int, default=-1, help='size of image width')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
    parser.add_argument('--model_epoch', type=int, default=185, help='epoch of trained model to load')
    parser.add_argument('--score_method', type=str, default='mse', choices=['mse', 'l1', 'combined'], help='anomaly scoring method')
    parser.add_argument('--threshold', type=float, default=None, help='anomaly threshold (if None, will be computed automatically)')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--border_margin', type=int, default=4, help='number of pixels to exclude from borders')
    parser.add_argument('--use_concentration', type=bool, default=True, help='whether to apply concentration penalty')
    parser.add_argument('--concentration_weight', type=float, default=2.0, help='weight for concentration penalty')
    parser.add_argument('--concentration_method', type=str, default='centroid', choices=['centroid', 'patch'], help='concentration method')
    parser.add_argument('--balance_dataset', action='store_true', help='balance the test dataset')
    parser.add_argument('--save_segmentation', action='store_true', help='save segmentation masks')
    parser.add_argument('--save_per_class', type=int, default=3, help='number of examples to save per class')
    
    opt = parser.parse_args()
    
    print("Multi-Class Abnormal-to-Normal Testing Configuration:")
    print(opt)
    
    # Create output directory
    output_dir = f'output/abnormal_to_normal/{opt.dataset_name}/multi_class_epoch_{opt.model_epoch}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean previous results
    print("Cleaning previous test results...")
    for pattern in ['*.png', '*.csv']:
        files_to_remove = glob.glob(os.path.join(output_dir, pattern))
        for file_path in files_to_remove:
            os.remove(file_path)
    
    print("Starting multi-class abnormal-to-normal evaluation...")
    
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = AbnormalToNormalDetector(opt.channels, opt.channels, opt.n_residual_blocks)
    if cuda:
        model = model.cuda()
    
    # Load trained model
    model_path = f'saved_models/{opt.dataset_name}/abnormal_to_normal_{opt.model_epoch}.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith('segmentation_head')}
        model.load_state_dict(filtered_checkpoint, strict=False)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model not found at {model_path}")
        exit(1)
    
    model.eval()
    
    # Image transformations
    transforms_ = []
    if opt.img_height != -1 and opt.img_width != -1:
        transforms_.append(transforms.Resize((opt.img_height, opt.img_width)))
    
    transforms_.extend([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create multi-class dataset
    print("\nLoading multi-class dataset...")
    all_data, class_counts = create_multi_class_dataset(opt.dataset_name, transforms_, opt.balance_dataset)
    
    if len(all_data) == 0:
        print("Error: No data loaded!")
        exit(1)
    
    dataset = MultiClassDataset(all_data)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    
    print(f"\nDataset loaded:")
    print(f"Total images: {len(dataset)}")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images")
    
    # Testing
    print("\nRunning multi-class evaluation...")
    
    all_scores = []
    all_binary_labels = []
    all_disease_classes = []
    all_images = []
    all_healed_images = []
    all_difference_maps = []
    all_segmentation_masks = []
    all_paths = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Get batch data
            images = Variable(batch['image']).to(device)
            binary_labels = batch['binary_label'].numpy()
            disease_classes = batch['disease_class']
            paths = batch['path']
            
            # Get model predictions
            results = model.detect_and_segment(
                images,
                threshold=None,
                border_margin=opt.border_margin,
                method=opt.score_method,
                use_concentration=opt.use_concentration,
                concentration_weight=opt.concentration_weight,
                concentration_method=opt.concentration_method
            )
            
            # Store results
            all_scores.extend(results['anomaly_scores'].cpu().numpy())
            all_binary_labels.extend(binary_labels)
            all_disease_classes.extend(disease_classes)
            all_images.extend(images.cpu())
            all_healed_images.extend(results['healed_images'].cpu())
            all_difference_maps.extend(results['difference_maps'].cpu())
            all_segmentation_masks.extend(results['segmentation_masks'].cpu())
            all_paths.extend(paths)
    
    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_binary_labels = np.array(all_binary_labels)
    all_disease_classes = np.array(all_disease_classes)
    
    print(f"\nProcessed {len(all_scores)} images")
    for class_name in DISEASE_CLASSES:
        count = np.sum(all_disease_classes == class_name)
        print(f"  {class_name}: {count} images")
    
    # Evaluate performance
    threshold, auc_score, class_results = evaluate_multi_class_performance(
        all_scores, all_binary_labels, all_disease_classes, output_dir
    )
    
    # Visualize results
    if opt.save_segmentation:
        visualize_multi_class_results(
            all_images, all_healed_images, all_difference_maps, all_segmentation_masks,
            all_scores, all_binary_labels, all_disease_classes, all_paths, output_dir, opt.save_per_class
        )
    
    # Save detailed results
    results_df = pd.DataFrame({
        'image_path': all_paths,
        'disease_class': all_disease_classes,
        'healing_difference_score': all_scores,
        'binary_label': all_binary_labels,
        'predicted_binary': (all_scores > threshold).astype(int)
    })
    
    results_df.to_csv(f'{output_dir}/multi_class_detailed_results.csv', index=False)
    
    # Save summary results
    summary_df = pd.DataFrame.from_dict(class_results, orient='index')
    summary_df.to_csv(f'{output_dir}/multi_class_summary.csv')
    
    print(f"\nResults saved to {output_dir}/")
    print("Files created:")
    print("  - multi_class_detailed_results.csv: Detailed results for each image")
    print("  - multi_class_summary.csv: Summary statistics per class")
    print("  - multi_class_roc_curve.png: ROC curve")
    print("  - multi_class_score_distribution.png: Score distributions by class")
    print("  - multi_class_boxplot.png: Box plot of scores by class")
    print("  - multi_class_performance.png: Performance bar chart")
    print("  - confusion_matrix.png: Confusion matrix")
    if opt.save_segmentation:
        print("  - [CLASS]_example_*.png: Example healing visualizations for each class")

if __name__ == '__main__':
    main()
