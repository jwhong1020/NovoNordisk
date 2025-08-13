#!/usr/bin/env python3
"""
Binary Classification Testing Script for Abnormal-to-Normal Generator
Tests Normal vs each disease class separately: CNV, DME, DRUSEN

Usage:
python test_binary_classification.py --dataset_name datasets/preprocessed_128 --model_epoch 185
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

def compute_optimal_threshold(scores, labels):
    """Compute optimal threshold using ROC curve"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Find threshold that maximizes (tpr - fpr)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

def evaluate_binary_classification(scores, labels, disease_name, output_dir, threshold=None):
    """Evaluate binary classification performance"""
    
    if threshold is None:
        threshold = compute_optimal_threshold(scores, labels)
    
    predictions = (scores > threshold).astype(int)
    
    # Compute metrics
    auc_score = roc_auc_score(labels, scores)
    
    # Classification report
    report = classification_report(labels, predictions, 
                                 target_names=['Normal', disease_name], 
                                 output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    print(f"\n{'='*60}")
    print(f"BINARY CLASSIFICATION: NORMAL vs {disease_name}")
    print(f"{'='*60}")
    print(f"Optimal Threshold: {threshold:.4f}")
    print(f"AUC-ROC Score: {auc_score:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['Normal', disease_name]))
    
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Normal  {disease_name}")
    print(f"Actual Normal    {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       {disease_name:6s}    {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"  Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  Specificity:          {specificity:.4f}")
    print(f"  Precision (PPV):      {precision:.4f}")
    print(f"  NPV:                  {npv:.4f}")
    print(f"  F1-Score:             {report[disease_name]['f1-score']:.4f}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC Curve: Normal vs {disease_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/roc_curve_{disease_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', disease_name],
                yticklabels=['Normal', disease_name])
    plt.title(f'Confusion Matrix: Normal vs {disease_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{output_dir}/confusion_matrix_{disease_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot score distribution
    plt.figure(figsize=(10, 6))
    normal_scores = scores[labels == 0]
    disease_scores = scores[labels == 1]
    
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True)
    plt.hist(disease_scores, bins=50, alpha=0.7, label=disease_name, density=True)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Healing Difference Score')
    plt.ylabel('Density')
    plt.title(f'Score Distribution: Normal vs {disease_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/score_distribution_{disease_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'disease': disease_name,
        'threshold': threshold,
        'auc': auc_score,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'f1_score': report[disease_name]['f1-score'],
        'accuracy': report['accuracy'],
        'confusion_matrix': cm,
        'n_samples': len(scores),
        'n_normal': np.sum(labels == 0),
        'n_disease': np.sum(labels == 1)
    }

def test_disease_class(model, dataset_name, disease_class, transforms_, opt, device):
    """Test model on a specific disease class vs normal"""
    
    print(f"\n{'='*60}")
    print(f"TESTING: NORMAL vs {disease_class}")
    print(f"{'='*60}")
    
    # Create dataset for this specific disease
    dataset = AnomalyDatasetGrayscale(
        normal_dir=f"{dataset_name}/test/NORMAL",
        abnormal_dir=f"{dataset_name}/test/{disease_class}",
        transform=transforms.Compose(transforms_),
        balance_dataset=opt.balance_dataset
    )
    
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    
    print(f"Dataset size: {len(dataset)} images")
    
    all_scores = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Processing {disease_class}")):
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
            all_paths.extend(paths)
    
    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    print(f"Normal images: {np.sum(all_labels == 0)}")
    print(f"{disease_class} images: {np.sum(all_labels == 1)}")
    
    return all_scores, all_labels, all_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='datasets/preprocessed_128', help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--img_height', type=int, default=-1, help='size of image height')
    parser.add_argument('--img_width', type=int, default=-1, help='size of image width')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
    parser.add_argument('--model_epoch', type=int, default=185, help='epoch of trained model to load')
    parser.add_argument('--score_method', type=str, default='mse', choices=['mse', 'l1', 'combined'], help='anomaly scoring method')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--border_margin', type=int, default=4, help='number of pixels to exclude from borders when calculating anomaly score')
    parser.add_argument('--use_concentration', type=bool, default=True, help='whether to apply concentration penalty for clustered anomalies')
    parser.add_argument('--concentration_weight', type=float, default=2.0, help='weight for concentration penalty')
    parser.add_argument('--concentration_method', type=str, default='centroid', choices=['centroid', 'patch'], help='method for computing concentration penalty')
    parser.add_argument('--balance_dataset', action='store_true', help='balance the test dataset by sampling equal numbers from normal and abnormal classes')
    
    opt = parser.parse_args()
    
    print("Binary Classification Testing Configuration:")
    print(opt)

    # Create output directory
    output_dir = f'output/binary_classification/{opt.dataset_name}/epoch_{opt.model_epoch}'
    os.makedirs(output_dir, exist_ok=True)

    # Clean previous results
    print("Cleaning previous test results...")
    try:
        for pattern in ['*.png', '*.csv']:
            files_to_remove = glob.glob(os.path.join(output_dir, pattern))
            for file_path in files_to_remove:
                os.remove(file_path)
        print("  Output directory cleaned")
    except Exception as e:
        print(f"  Error cleaning directory: {e}")

    # Device setup
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
        # Filter out segmentation head parameters if they exist
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

    # Disease classes to test
    disease_classes = ['CNV', 'DME', 'DRUSEN']
    
    # Store results for comparison
    all_results = []
    comparison_data = []
    
    # Test each disease class separately
    for disease_class in disease_classes:
        print(f"\n{'#'*80}")
        print(f"TESTING DISEASE CLASS: {disease_class}")
        print(f"{'#'*80}")
        
        try:
            # Test this disease class
            scores, labels, paths = test_disease_class(
                model, opt.dataset_name, disease_class, transforms_, opt, device
            )
            
            # Evaluate performance
            result = evaluate_binary_classification(
                scores, labels, disease_class, output_dir
            )
            all_results.append(result)
            
            # Store detailed results
            disease_df = pd.DataFrame({
                'image_path': paths,
                'healing_difference_score': scores,
                'true_label': labels,
                'predicted_label': (scores > result['threshold']).astype(int),
                'disease_class': disease_class
            })
            disease_df.to_csv(f'{output_dir}/detailed_results_{disease_class.lower()}.csv', index=False)
            
            # Add to comparison data
            for i, (score, label, path) in enumerate(zip(scores, labels, paths)):
                comparison_data.append({
                    'disease_class': disease_class,
                    'image_path': path,
                    'healing_score': score,
                    'true_label': 'Normal' if label == 0 else disease_class,
                    'predicted_label': 'Normal' if score <= result['threshold'] else disease_class
                })
                
        except Exception as e:
            print(f"Error testing {disease_class}: {e}")
            continue
    
    # Create comparison summary
    print(f"\n{'='*80}")
    print("BINARY CLASSIFICATION SUMMARY - ALL DISEASES")
    print(f"{'='*80}")
    
    if all_results:
        # Summary table
        summary_df = pd.DataFrame([
            {
                'Disease': result['disease'],
                'AUC-ROC': f"{result['auc']:.4f}",
                'Sensitivity': f"{result['sensitivity']:.4f}",
                'Specificity': f"{result['specificity']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'F1-Score': f"{result['f1_score']:.4f}",
                'Accuracy': f"{result['accuracy']:.4f}",
                'Threshold': f"{result['threshold']:.4f}",
                'N_Samples': result['n_samples'],
                'N_Normal': result['n_normal'],
                'N_Disease': result['n_disease']
            }
            for result in all_results
        ])
        
        print(summary_df.to_string(index=False))
        summary_df.to_csv(f'{output_dir}/binary_classification_summary.csv', index=False)
        
        # Comparison plot - AUC scores
        plt.figure(figsize=(10, 6))
        diseases = [r['disease'] for r in all_results]
        aucs = [r['auc'] for r in all_results]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = plt.bar(diseases, aucs, color=colors, alpha=0.7, edgecolor='black')
        plt.ylabel('AUC-ROC Score')
        plt.title('Binary Classification Performance: Normal vs Each Disease')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, auc in zip(bars, aucs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/auc_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Metrics comparison heatmap
        metrics_data = []
        for result in all_results:
            metrics_data.append([
                result['auc'],
                result['sensitivity'], 
                result['specificity'],
                result['precision'],
                result['f1_score']
            ])
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(metrics_data, 
                   annot=True, 
                   fmt='.3f',
                   cmap='YlOrRd',
                   xticklabels=['AUC-ROC', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score'],
                   yticklabels=[r['disease'] for r in all_results],
                   cbar_kws={'label': 'Score'})
        plt.title('Binary Classification Metrics Comparison')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save all comparison data
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(f'{output_dir}/all_diseases_comparison.csv', index=False)
        
        print(f"\n{'='*60}")
        print("FILES CREATED:")
        print(f"{'='*60}")
        print(f"📊 Summary:")
        print(f"  - binary_classification_summary.csv")
        print(f"  - auc_comparison.png")
        print(f"  - metrics_heatmap.png")
        print(f"  - all_diseases_comparison.csv")
        
        print(f"\n📈 Per Disease:")
        for disease in diseases:
            print(f"  {disease}:")
            print(f"    - detailed_results_{disease.lower()}.csv")
            print(f"    - roc_curve_{disease.lower()}.png")
            print(f"    - confusion_matrix_{disease.lower()}.png")
            print(f"    - score_distribution_{disease.lower()}.png")
        
        print(f"\n📍 Results saved to: {output_dir}/")
        
        # Best performing disease
        best_disease = max(all_results, key=lambda x: x['auc'])
        print(f"\n🏆 Best Performance: Normal vs {best_disease['disease']} (AUC: {best_disease['auc']:.4f})")
    
    else:
        print("❌ No results generated. Check if disease directories exist in the dataset.")
