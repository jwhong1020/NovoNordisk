#!/usr/bin/env python3
"""
Evaluate all saved abnormal-to-normal models and record performance metrics to CSV.
This script tests all models with pattern 'abnormal_to_normal_*.pth' and saves
only the key metrics without generating any visualization files.
"""

import os
import re
import csv
import torch
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from datetime import datetime
from torchvision import transforms

# Import existing modules
from abnormal_to_normal_model import AbnormalToNormalDetector
from anomaly_datasets import AnomalyDatasetGrayscale
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, accuracy_score, f1_score


def get_model_epochs(models_dir):
    """Get all available model epochs from the saved models directory."""
    epochs = []
    pattern = re.compile(r'abnormal_to_normal_(\d+)\.pth')
    
    for filename in os.listdir(models_dir):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            epochs.append(epoch)
    
    return sorted(epochs)


def evaluate_model(model_path, dataset_path, device, balance_dataset=True, 
                  border_margin=4, use_concentration=True, concentration_weight=2.0):
    """
    Evaluate a single model and return metrics.
    """
    try:
        # Load model
        model = AbnormalToNormalDetector(
            input_nc=1, output_nc=1, n_residual_blocks=9
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Filter out segmentation head parameters if they exist
        filtered_checkpoint = {}
        for key, value in checkpoint.items():
            if not key.startswith('segmentation_head'):
                filtered_checkpoint[key] = value
        
        model.load_state_dict(filtered_checkpoint, strict=False)
        model.eval()
        
        # Load dataset
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
        
        test_dataset = AnomalyDatasetGrayscale(
            normal_dir=f"{dataset_path}/test/NORMAL",
            abnormal_dir=f"{dataset_path}/test/CNV",  # Using CNV as abnormal class
            transform=transform,
            balance_dataset=balance_dataset
        )
        
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        # Run inference
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                labels = batch['label']
                
                results = model.detect_and_segment(
                    images, threshold=None, 
                    border_margin=border_margin,
                    use_concentration=use_concentration,
                    concentration_weight=concentration_weight
                )
                
                scores = results['anomaly_scores'].cpu().numpy()
                all_scores.extend(scores)
                all_labels.extend(labels.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        auc_roc = roc_auc_score(all_labels, all_scores)
        
        # Find optimal threshold using ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Get predictions with optimal threshold
        predictions = (all_scores > optimal_threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, predictions)
        f1 = f1_score(all_labels, predictions)
        
        # Get precision and recall from classification report
        report = classification_report(all_labels, predictions, output_dict=True)
        precision = report['1']['precision']  # Anomaly class precision
        recall = report['1']['recall']  # Anomaly class recall
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'precision': precision,
            'recall': recall,
            'optimal_threshold': optimal_threshold,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'accuracy': None,
            'f1_score': None,
            'auc_roc': None,
            'precision': None,
            'recall': None,
            'optimal_threshold': None,
            'status': f'error: {str(e)}'
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate all saved abnormal-to-normal models')
    parser.add_argument('--dataset_name', type=str, default='datasets/preprocessed_128',
                       help='Dataset directory name')
    parser.add_argument('--models_dir', type=str, default='saved_models/datasets/preprocessed_128',
                       help='Directory containing saved models')
    parser.add_argument('--output_csv', type=str, default='model_evaluation_results.csv',
                       help='Output CSV filename')
    parser.add_argument('--balance_dataset', action='store_true', default=True,
                       help='Balance the dataset classes')
    parser.add_argument('--border_margin', type=int, default=4,
                       help='Border margin for anomaly scoring')
    parser.add_argument('--use_concentration', action='store_true', default=True,
                       help='Use concentration penalty')
    parser.add_argument('--concentration_weight', type=float, default=2.0,
                       help='Concentration penalty weight')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get all available model epochs
    epochs = get_model_epochs(args.models_dir)
    print(f"Found {len(epochs)} models to evaluate (epochs {min(epochs)}-{max(epochs)})")
    
    # Prepare CSV file
    csv_path = args.output_csv
    fieldnames = ['epoch', 'accuracy', 'f1_score', 'auc_roc', 'precision', 'recall', 
                  'optimal_threshold', 'status', 'evaluation_time']
    
    # Write CSV header
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    print(f"Starting evaluation of {len(epochs)} models...")
    print(f"Results will be saved to: {csv_path}")
    
    # Evaluate each model
    for epoch in tqdm(epochs, desc="Evaluating models"):
        model_path = os.path.join(args.models_dir, f'abnormal_to_normal_{epoch}.pth')
        
        if not os.path.exists(model_path):
            continue
            
        start_time = datetime.now()
        
        # Evaluate model
        metrics = evaluate_model(
            model_path=model_path,
            dataset_path=args.dataset_name,
            device=device,
            balance_dataset=args.balance_dataset,
            border_margin=args.border_margin,
            use_concentration=args.use_concentration,
            concentration_weight=args.concentration_weight
        )
        
        end_time = datetime.now()
        evaluation_time = (end_time - start_time).total_seconds()
        
        # Add epoch and evaluation time
        metrics['epoch'] = epoch
        metrics['evaluation_time'] = round(evaluation_time, 2)
        
        # Append to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(metrics)
        
        # Print progress for successful evaluations
        if metrics['status'] == 'success':
            print(f"Epoch {epoch:3d}: AUC={metrics['auc_roc']:.4f}, "
                  f"ACC={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        else:
            print(f"Epoch {epoch:3d}: {metrics['status']}")
    
    print(f"\nEvaluation complete! Results saved to {csv_path}")
    
    # Print summary statistics
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        successful_runs = df[df['status'] == 'success']
        
        if len(successful_runs) > 0:
            print(f"\nSummary Statistics ({len(successful_runs)} successful evaluations):")
            print(f"Best AUC-ROC: {successful_runs['auc_roc'].max():.4f} (Epoch {successful_runs.loc[successful_runs['auc_roc'].idxmax(), 'epoch']})")
            print(f"Best Accuracy: {successful_runs['accuracy'].max():.4f} (Epoch {successful_runs.loc[successful_runs['accuracy'].idxmax(), 'epoch']})")
            print(f"Best F1-Score: {successful_runs['f1_score'].max():.4f} (Epoch {successful_runs.loc[successful_runs['f1_score'].idxmax(), 'epoch']})")
            print(f"Average AUC-ROC: {successful_runs['auc_roc'].mean():.4f} ± {successful_runs['auc_roc'].std():.4f}")
            print(f"Average Accuracy: {successful_runs['accuracy'].mean():.4f} ± {successful_runs['accuracy'].std():.4f}")
            print(f"Average F1-Score: {successful_runs['f1_score'].mean():.4f} ± {successful_runs['f1_score'].std():.4f}")
        else:
            print("No successful evaluations found.")
            
    except ImportError:
        print("Install pandas for summary statistics: pip install pandas")


if __name__ == "__main__":
    main()
