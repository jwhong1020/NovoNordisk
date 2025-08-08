import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

def visualize_anomaly_heatmap(original_image, difference_map, save_path=None, alpha=0.6):
    """
    Create a heatmap overlay showing anomaly regions.
    
    Args:
        original_image: Original input image (tensor or numpy array)
        difference_map: Difference map from reconstruction (tensor or numpy array)
        save_path: Path to save the visualization
        alpha: Transparency of the heatmap overlay
    """
    # Convert tensors to numpy if needed
    if torch.is_tensor(original_image):
        original_image = original_image.cpu().numpy()
    if torch.is_tensor(difference_map):
        difference_map = difference_map.cpu().numpy()
    
    # Normalize images to [0, 1]
    if original_image.max() > 1.0:
        original_image = original_image / 255.0
    if difference_map.max() > 1.0:
        difference_map = difference_map / 255.0
    
    # Convert from CHW to HWC format if needed
    if original_image.shape[0] == 3:
        original_image = np.transpose(original_image, (1, 2, 0))
    if difference_map.shape[0] == 3:
        difference_map = np.transpose(difference_map, (1, 2, 0))
    
    # Convert to grayscale if needed
    if len(difference_map.shape) == 3:
        difference_map = np.mean(difference_map, axis=2)
    
    # Create heatmap
    heatmap = cv2.applyColorMap((difference_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Create overlay
    overlay = alpha * heatmap + (1 - alpha) * original_image
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(difference_map, cmap='hot')
    axes[1].set_title('Anomaly Map')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_anomaly_statistics(anomaly_scores, labels):
    """
    Compute comprehensive statistics for anomaly detection results.
    
    Args:
        anomaly_scores: Array of anomaly scores
        labels: Array of true labels (0=normal, 1=anomaly)
    
    Returns:
        Dictionary with various statistics
    """
    normal_scores = anomaly_scores[labels == 0]
    anomaly_scores_subset = anomaly_scores[labels == 1]
    
    stats = {
        'normal_mean': np.mean(normal_scores),
        'normal_std': np.std(normal_scores),
        'normal_min': np.min(normal_scores),
        'normal_max': np.max(normal_scores),
        'anomaly_mean': np.mean(anomaly_scores_subset),
        'anomaly_std': np.std(anomaly_scores_subset),
        'anomaly_min': np.min(anomaly_scores_subset),
        'anomaly_max': np.max(anomaly_scores_subset),
        'separation': np.mean(anomaly_scores_subset) - np.mean(normal_scores),
        'overlap_ratio': len(normal_scores[normal_scores > np.mean(anomaly_scores_subset)]) / len(normal_scores)
    }
    
    return stats


def find_optimal_threshold_methods(scores, labels):
    """
    Find optimal threshold using multiple methods.
    
    Args:
        scores: Anomaly scores
        labels: True labels
    
    Returns:
        Dictionary with thresholds from different methods
    """
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    # Method 1: ROC curve (Youden's J statistic)
    fpr, tpr, thresholds_roc = roc_curve(labels, scores)
    youden_j = tpr - fpr
    optimal_idx_roc = np.argmax(youden_j)
    threshold_roc = thresholds_roc[optimal_idx_roc]
    
    # Method 2: Precision-Recall curve (F1-score)
    precision, recall, thresholds_pr = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx_pr = np.argmax(f1_scores)
    threshold_pr = thresholds_pr[optimal_idx_pr] if optimal_idx_pr < len(thresholds_pr) else thresholds_pr[-1]
    
    # Method 3: Statistical (mean + k*std of normal samples)
    normal_scores = scores[labels == 0]
    threshold_statistical = np.mean(normal_scores) + 2 * np.std(normal_scores)
    
    # Method 4: Percentile-based (95th percentile of normal samples)
    threshold_percentile = np.percentile(normal_scores, 95)
    
    return {
        'roc_optimal': threshold_roc,
        'pr_optimal': threshold_pr,
        'statistical': threshold_statistical,
        'percentile_95': threshold_percentile
    }


def create_anomaly_report(scores, labels, save_path=None):
    """
    Create a comprehensive anomaly detection report.
    
    Args:
        scores: Anomaly scores
        labels: True labels
        save_path: Path to save the report
    """
    from sklearn.metrics import classification_report, roc_auc_score
    
    # Get statistics
    stats = compute_anomaly_statistics(scores, labels)
    
    # Get optimal thresholds
    thresholds = find_optimal_threshold_methods(scores, labels)
    
    # Compute AUC
    auc = roc_auc_score(labels, scores)
    
    # Create report
    report = f"""
ANOMALY DETECTION REPORT
{'='*50}

Dataset Statistics:
- Total samples: {len(scores)}
- Normal samples: {np.sum(labels == 0)}
- Anomalous samples: {np.sum(labels == 1)}
- Anomaly ratio: {np.sum(labels == 1) / len(labels):.2%}

Score Statistics:
Normal Images:
- Mean: {stats['normal_mean']:.4f}
- Std: {stats['normal_std']:.4f}
- Min: {stats['normal_min']:.4f}
- Max: {stats['normal_max']:.4f}

Anomalous Images:
- Mean: {stats['anomaly_mean']:.4f}
- Std: {stats['anomaly_std']:.4f}
- Min: {stats['anomaly_min']:.4f}
- Max: {stats['anomaly_max']:.4f}

Separation Metrics:
- Score separation: {stats['separation']:.4f}
- Overlap ratio: {stats['overlap_ratio']:.2%}
- AUC-ROC: {auc:.4f}

Optimal Thresholds:
- ROC-based: {thresholds['roc_optimal']:.4f}
- PR-based: {thresholds['pr_optimal']:.4f}
- Statistical (μ+2σ): {thresholds['statistical']:.4f}
- 95th percentile: {thresholds['percentile_95']:.4f}

"""
    
    # Add classification results for each threshold
    for method, threshold in thresholds.items():
        predictions = (scores > threshold).astype(int)
        report += f"\nResults with {method} threshold ({threshold:.4f}):\n"
        report += classification_report(labels, predictions, target_names=['Normal', 'Anomaly'])
        report += "\n" + "-"*50 + "\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {save_path}")
    else:
        print(report)
    
    return report


def preprocess_medical_image(image_path, target_size=(256, 256)):
    """
    Preprocess medical images for anomaly detection.
    
    Args:
        image_path: Path to the medical image
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image tensor
    """
    import torchvision.transforms as transforms
    
    # Define preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def batch_inference(model, image_folder, output_folder, threshold=None):
    """
    Run batch inference on a folder of images.
    
    Args:
        model: Trained anomaly detection model
        image_folder: Folder containing images to process
        output_folder: Folder to save results
        threshold: Anomaly threshold
    """
    import glob
    from torchvision.utils import save_image
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))
    
    print(f"Found {len(image_files)} images to process")
    
    results = []
    model.eval()
    
    with torch.no_grad():
        for i, image_path in enumerate(image_files):
            try:
                # Preprocess image
                image_tensor = preprocess_medical_image(image_path)
                
                if torch.cuda.is_available():
                    image_tensor = image_tensor.cuda()
                
                # Run inference
                result = model.detect_anomalies(image_tensor, threshold)
                
                # Save results
                filename = os.path.splitext(os.path.basename(image_path))[0]
                
                # Save reconstructed image
                save_image(result['reconstructed'], 
                          os.path.join(output_folder, f"{filename}_reconstructed.png"),
                          normalize=True)
                
                # Save difference map
                save_image(result['difference_maps'], 
                          os.path.join(output_folder, f"{filename}_difference.png"),
                          normalize=True)
                
                # Create heatmap visualization
                visualize_anomaly_heatmap(
                    image_tensor[0], 
                    result['difference_maps'][0],
                    os.path.join(output_folder, f"{filename}_heatmap.png")
                )
                
                # Store results
                results.append({
                    'image_path': image_path,
                    'anomaly_score': result['anomaly_scores'][0].item(),
                    'prediction': result['predictions'][0].item() if result['predictions'] is not None else None
                })
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_files)} images")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
    
    # Save results summary
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, 'batch_results.csv'), index=False)
    
    print(f"Batch inference completed. Results saved to {output_folder}")
    return results_df
