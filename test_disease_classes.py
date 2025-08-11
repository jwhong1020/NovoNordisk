#!/usr/bin/env python3
"""
Test script to evaluate anomaly detection performance across different disease classes.
Tests Normal vs CNV, Normal vs DME, Normal vs DRUSEN individually.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from anomaly_models import AnomalyDetector
from anomaly_datasets import MultiClassAnomalyDatasetGrayscale
from utils import tensor2image

def test_disease_class(anomaly_detector, dataset_name, disease_class, opt):
    """Test performance on a specific disease class"""
    
    print(f"\n{'='*60}")
    print(f"TESTING: NORMAL vs {disease_class}")
    print(f"{'='*60}")
    
    # Image transformations for grayscale
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
    
    # Create dataset for this disease class
    dataset = MultiClassAnomalyDatasetGrayscale(
        dataset_root=dataset_name,
        disease_class=disease_class,
        transform=transforms.Compose(transforms_)
    )
    
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    
    # Run inference
    all_scores = []
    all_labels = []
    all_paths = []
    
    print(f"Running inference on {len(dataset)} images...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            real_images = Variable(batch['image'].type(Tensor))
            labels = batch['label'].numpy()
            paths = batch['path']
            
            # Detect anomalies with concentration penalty
            results = anomaly_detector.detect_anomalies(
                real_images, 
                border_margin=opt.border_margin,
                use_concentration=opt.use_concentration,
                concentration_weight=opt.concentration_weight,
                concentration_method=opt.concentration_method
            )
            scores = results['anomaly_scores'].cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels)
            all_paths.extend(paths)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(dataloader)} batches")
    
    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    if disease_class != 'ALL':
        # Binary classification (Normal vs Disease)
        binary_labels = (all_labels > 0).astype(int)  # Convert to binary: 0=normal, 1=disease
        auc_score = roc_auc_score(binary_labels, all_scores)
        
        # Compute optimal threshold
        fpr, tpr, thresholds = roc_curve(binary_labels, all_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        predictions = (all_scores > optimal_threshold).astype(int)
        
        print(f"\nRESULTS for {disease_class}:")
        print(f"  AUC-ROC Score: {auc_score:.4f}")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  Accuracy: {np.mean(predictions == binary_labels):.4f}")
        
        # Classification report
        report = classification_report(binary_labels, predictions, 
                                     target_names=['Normal', disease_class], 
                                     output_dict=True)
        print("\nClassification Report:")
        print(classification_report(binary_labels, predictions, target_names=['Normal', disease_class]))
        
        # Get top anomalies for this disease class
        disease_indices = np.where(all_labels > 0)[0]  # Get disease samples
        disease_scores = all_scores[disease_indices]
        disease_paths = [all_paths[i] for i in disease_indices]
        
        # Sort by score (descending)
        sorted_indices = np.argsort(disease_scores)[::-1]
        
        print(f"\nTop 5 {disease_class} detections:")
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[i]
            score = disease_scores[idx]
            path = disease_paths[idx]
            print(f"  {i+1}. Score: {score:.4f}, File: {os.path.basename(path)}")
        
        return {
            'disease_class': disease_class,
            'auc_roc': auc_score,
            'threshold': optimal_threshold,
            'accuracy': np.mean(predictions == binary_labels),
            'report': report,
            'normal_count': np.sum(all_labels == 0),
            'disease_count': np.sum(all_labels > 0)
        }
    
    return None

def create_comparison_plots(results, output_dir):
    """Create comparison plots across disease classes"""
    
    # Extract metrics
    disease_classes = [r['disease_class'] for r in results]
    auc_scores = [r['auc_roc'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # AUC-ROC comparison
    bars1 = ax1.bar(disease_classes, auc_scores, color=['#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('AUC-ROC Score')
    ax1.set_title('Anomaly Detection Performance by Disease Class')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars1, auc_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy comparison
    bars2 = ax2.bar(disease_classes, accuracies, color=['#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Classification Accuracy by Disease Class')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars2, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'disease_class_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create detailed results table
    results_df = pd.DataFrame([
        {
            'Disease Class': r['disease_class'],
            'AUC-ROC': f"{r['auc_roc']:.4f}",
            'Accuracy': f"{r['accuracy']:.4f}",
            'Threshold': f"{r['threshold']:.4f}",
            'Normal Count': r['normal_count'],
            'Disease Count': r['disease_count'],
            'Precision (Disease)': f"{r['report'][r['disease_class']]['precision']:.4f}",
            'Recall (Disease)': f"{r['report'][r['disease_class']]['recall']:.4f}",
            'F1-Score (Disease)': f"{r['report'][r['disease_class']]['f1-score']:.4f}"
        }
        for r in results
    ])
    
    results_df.to_csv(os.path.join(output_dir, 'disease_class_results.csv'), index=False)
    print(f"\nDetailed results saved to {os.path.join(output_dir, 'disease_class_results.csv')}")
    
    return results_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='OCT2017', help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--img_height', type=int, default=256, help='size of image height')
    parser.add_argument('--img_width', type=int, default=256, help='size of image width')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels (1 for grayscale)')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
    parser.add_argument('--model_epoch', type=int, default=1, help='epoch of trained model to load')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--border_margin', type=int, default=16, help='number of pixels to exclude from borders')
    parser.add_argument('--use_concentration', type=bool, default=True, help='whether to apply concentration penalty')
    parser.add_argument('--concentration_weight', type=float, default=2.0, help='weight for concentration penalty')
    parser.add_argument('--concentration_method', type=str, default='centroid', choices=['centroid', 'patch'], help='concentration method')
    parser.add_argument('--disease_classes', type=str, nargs='+', default=['CNV', 'DME', 'DRUSEN'], 
                        help='disease classes to test (default: CNV DME DRUSEN)')
    opt = parser.parse_args()
    print(opt)

    # Create output directory
    output_dir = f'output/disease_class_analysis/{opt.dataset_name}'
    os.makedirs(output_dir, exist_ok=True)

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Initialize and load model
    anomaly_detector = AnomalyDetector(opt.channels, opt.channels, opt.n_residual_blocks)
    if cuda:
        anomaly_detector = anomaly_detector.cuda()

    model_path = f'saved_models/{opt.dataset_name}/anomaly_detector_gray_{opt.model_epoch}.pth'
    if os.path.exists(model_path):
        anomaly_detector.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}")
        exit(1)

    anomaly_detector.eval()

    # Test each disease class
    all_results = []
    for disease_class in opt.disease_classes:
        result = test_disease_class(anomaly_detector, opt.dataset_name, disease_class, opt)
        if result:
            all_results.append(result)

    # Create comparison analysis
    if all_results:
        print(f"\n{'='*60}")
        print("DISEASE CLASS COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        results_df = create_comparison_plots(all_results, output_dir)
        print(results_df.to_string(index=False))
        
        # Find best and worst performing classes
        best_class = max(all_results, key=lambda x: x['auc_roc'])
        worst_class = min(all_results, key=lambda x: x['auc_roc'])
        
        print(f"\n🏆 BEST PERFORMANCE: {best_class['disease_class']} (AUC-ROC: {best_class['auc_roc']:.4f})")
        print(f"📉 WORST PERFORMANCE: {worst_class['disease_class']} (AUC-ROC: {worst_class['auc_roc']:.4f})")
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
