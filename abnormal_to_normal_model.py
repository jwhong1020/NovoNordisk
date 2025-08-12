#!/usr/bin/env python3
"""
Abnormal-to-Normal Generator for Anomaly Detection and Segmentation

This model learns to "heal" abnormal images to look normal.
Training Strategy:
1. Normal images: Generator should output identical image (identity loss)
2. Abnormal images: Generator should output "healed" normal version
3. Difference map between input and output reveals anomaly locations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from models import Generator, Discriminator

class AbnormalToNormalDetector(nn.Module):
    """
    Abnormal-to-Normal Generator for Anomaly Detection and Segmentation
    """
    
    def __init__(self, input_nc=1, output_nc=1, n_residual_blocks=9):
        super(AbnormalToNormalDetector, self).__init__()
        
        # Main generator: Input → "Healed" output
        self.healing_generator = Generator(input_nc, output_nc, n_residual_blocks)
        
        # Discriminator: Ensures healed images look realistic and normal
        self.discriminator = Discriminator(output_nc)
    
    def forward(self, x):
        """Forward pass returns healed image"""
        return self.healing_generator(x)
    
    def get_segmentation_mask(self, input_img, healed_img, anomaly_score=None, base_threshold_percentile=85, 
                             min_cluster_size=20, max_threshold_multiplier=2.0, adaptive_threshold=True):
        """
        Generate segmentation mask from difference map using adaptive thresholding and noise removal.
        
        Args:
            input_img: Original input image
            healed_img: Healed output image  
            anomaly_score: Anomaly score for the image (if Normal, return empty mask)
            base_threshold_percentile: Starting percentile for threshold (default 85)
            min_cluster_size: Minimum size of connected components to keep
            max_threshold_multiplier: Maximum factor to multiply base threshold
            adaptive_threshold: Whether to use adaptive thresholding based on distribution
        """
        batch_size = input_img.shape[0]
        device = input_img.device
        h, w = input_img.shape[-2], input_img.shape[-1]
        
        # Initialize output masks
        segmentation_masks = torch.zeros(batch_size, 1, h, w, device=device)
        
        # Compute difference map
        diff_map = torch.abs(input_img - healed_img)
        
        # Process each image in the batch
        for i in range(batch_size):
            # Convert to numpy for processing
            diff_np = diff_map[i, 0].cpu().numpy()
            
            # Skip if difference map is essentially zero
            if diff_np.max() <= 1e-6:
                print(f"Batch {i}: Skipping - difference map is essentially zero")
                continue
            
            # Calculate base threshold
            base_threshold = np.percentile(diff_np, base_threshold_percentile)
            
            if adaptive_threshold:
                # Analyze the distribution to determine if there are clear anomalies
                diff_values = diff_np.flatten()
                
                # Calculate statistics
                mean_val = np.mean(diff_values)
                std_val = np.std(diff_values)
                
                # Use Otsu-like approach: find threshold that maximizes between-class variance
                max_threshold = base_threshold * max_threshold_multiplier
                n_thresholds = 20
                thresholds = np.linspace(base_threshold, max_threshold, n_thresholds)
                
                best_threshold = base_threshold
                best_variance = 0
                
                for thresh in thresholds:
                    # Calculate between-class variance
                    mask = diff_values > thresh
                    if np.sum(mask) == 0 or np.sum(mask) == len(diff_values):
                        continue
                        
                    w1 = np.sum(mask) / len(diff_values)
                    w2 = 1 - w1
                    
                    if w1 > 0 and w2 > 0:
                        mu1 = np.mean(diff_values[mask])
                        mu2 = np.mean(diff_values[~mask])
                        between_variance = w1 * w2 * (mu1 - mu2) ** 2
                        
                        if between_variance > best_variance:
                            best_variance = between_variance
                            best_threshold = thresh
                
                threshold_val = best_threshold
                
                # Additional check: if very few pixels exceed even the base threshold,
                # the image might be mostly normal
                high_diff_ratio = np.sum(diff_values > base_threshold) / len(diff_values)
                if high_diff_ratio < 0.05:  # Less than 5% high-difference pixels
                    threshold_val = base_threshold * 1.5  # Be more conservative
                    
            else:
                threshold_val = base_threshold
            
            # Create initial binary mask
            binary_mask = (diff_np > threshold_val).astype(np.uint8)
            
            # Remove small isolated components (noise reduction)
            if min_cluster_size > 0:
                binary_mask = self.remove_small_components_cv2(binary_mask, min_cluster_size)
            
            # Debug print
            print(f"Batch {i}: Diff range: [{diff_np.min():.6f}, {diff_np.max():.6f}], Threshold: {threshold_val:.6f} (adaptive: {adaptive_threshold})")
            white_pixels = np.sum(binary_mask)
            total_pixels = binary_mask.size
            percentage = white_pixels/total_pixels*100 if total_pixels > 0 else 0
            print(f"Batch {i}: Mask has {int(white_pixels)} white pixels out of {total_pixels} total ({percentage:.1f}%)")
            
            # Skip if no pixels survive the threshold after filtering
            if binary_mask.sum() == 0:
                print(f"Batch {i}: Skipping mask - no pixels above threshold after filtering")
                continue
            
            # Convert back to tensor (keep as 0-1 values)
            segmentation_masks[i, 0] = torch.from_numpy(binary_mask).float().to(device)
        
        return segmentation_masks
    
    def remove_small_components_cv2(self, binary_mask, min_size):
        """
        Remove small connected components from binary mask using OpenCV.
        Also applies morphological operations to clean up the mask.
        """
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        # Create output mask
        filtered_mask = np.zeros_like(binary_mask)
        
        # Keep only components larger than min_size
        for label in range(1, num_labels):  # Skip background (label 0)
            component_mask = (labels == label)
            if np.sum(component_mask) >= min_size:
                filtered_mask[component_mask] = 1
        
        # Apply morphological operations to clean up the mask
        # Close small gaps within anomaly regions
        kernel_close = np.ones((3, 3), np.uint8)
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Remove very small noise that might have been introduced by closing
        kernel_open = np.ones((2, 2), np.uint8)
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel_open)
        
        return filtered_mask
    
    def compute_anomaly_score(self, original, healed, method='mse', border_margin=16, 
                            use_concentration=True, concentration_weight=2.0, concentration_method='centroid'):
        """
        Compute anomaly score based on healing difference
        
        For normal images: Should have low difference (identity mapping)
        For abnormal images: Should have high difference (healing changes)
        """
        if method == 'mse':
            difference_map = torch.pow(original - healed, 2)
        elif method == 'l1':
            difference_map = torch.abs(original - healed)
        elif method == 'combined':
            mse_loss = torch.pow(original - healed, 2)
            l1_loss = torch.abs(original - healed)
            difference_map = 0.7 * mse_loss + 0.3 * l1_loss
        
        # Apply border exclusion
        if border_margin > 0:
            h, w = difference_map.shape[-2], difference_map.shape[-1]
            if h > 2 * border_margin and w > 2 * border_margin:
                difference_map = difference_map[:, :, border_margin:-border_margin, border_margin:-border_margin]
        
        # Calculate base anomaly score
        base_score = torch.mean(difference_map, dim=[1, 2, 3])
        
        # Apply concentration penalty if enabled
        if use_concentration and concentration_weight > 0:
            concentration_penalty = self.compute_concentration_penalty(
                difference_map, method=concentration_method
            )
            anomaly_score = base_score + concentration_weight * concentration_penalty
        else:
            anomaly_score = base_score
        
        return anomaly_score, difference_map
    
    def compute_concentration_penalty(self, difference_map, threshold_percentile=90, method='centroid'):
        """
        Compute concentration penalty that heavily penalizes large patches of concentrated differences.
        """
        batch_size = difference_map.shape[0]
        concentration_scores = []
        
        for i in range(batch_size):
            diff_map = difference_map[i, 0]  # Assume single channel
            
            # Create binary mask of high-difference pixels
            threshold = torch.quantile(diff_map.flatten(), threshold_percentile / 100.0)
            binary_mask = (diff_map > threshold).float()
            
            if method == 'centroid':
                # Spatial concentration using distance from center of mass
                if binary_mask.sum() > 0:
                    # Find center of mass of anomalous pixels
                    y_coords, x_coords = torch.meshgrid(
                        torch.arange(diff_map.shape[0], device=diff_map.device),
                        torch.arange(diff_map.shape[1], device=diff_map.device), 
                        indexing='ij'
                    )
                    
                    # Center of mass
                    total_mass = binary_mask.sum()
                    centroid_y = (binary_mask * y_coords).sum() / total_mass
                    centroid_x = (binary_mask * x_coords).sum() / total_mass
                    
                    # Calculate distances from centroid
                    distances = torch.sqrt((y_coords - centroid_y)**2 + (x_coords - centroid_x)**2)
                    
                    # Concentration score: inverse of average distance (higher = more concentrated)
                    avg_distance = (binary_mask * distances).sum() / total_mass
                    concentration_score = 1.0 / (avg_distance + 1e-6)
                else:
                    concentration_score = 0.0
                    
            elif method == 'patch':
                # Use max pooling to find largest connected components
                kernel_size = 5
                pooled = F.max_pool2d(binary_mask.unsqueeze(0), kernel_size, stride=1, padding=kernel_size//2)
                concentration_score = pooled.max().item()
            
            concentration_scores.append(concentration_score)
        
        return torch.tensor(concentration_scores, device=difference_map.device)
    
    def detect_and_segment(self, images, threshold=None, border_margin=16, method='mse',
                          use_concentration=True, concentration_weight=2.0, concentration_method='centroid'):
        """
        Detect anomalies and generate segmentation masks
        
        Returns:
            anomaly_scores: Scalar scores for classification
            healed_images: "Normal" versions of input images
            difference_maps: Pixel-wise difference maps
            segmentation_masks: Direct segmentation predictions
        """
        self.eval()
        with torch.no_grad():
            # Generate healed images
            healed_images = self.healing_generator(images)
            
            # Compute anomaly scores and difference maps
            anomaly_scores, difference_maps = self.compute_anomaly_score(
                images, healed_images, method=method, border_margin=border_margin,
                use_concentration=use_concentration, concentration_weight=concentration_weight,
                concentration_method=concentration_method
            )
            
            # Generate segmentation masks (using anomaly scores to skip normal images)
            segmentation_masks = self.get_segmentation_mask(
                images, healed_images, anomaly_scores, 
                base_threshold_percentile=85, min_cluster_size=20, adaptive_threshold=True
            )
            
            predictions = None
            if threshold is not None:
                predictions = (anomaly_scores > threshold).float()
            
            return {
                'anomaly_scores': anomaly_scores,
                'healed_images': healed_images,
                'difference_maps': difference_maps,
                'segmentation_masks': segmentation_masks,
                'predictions': predictions
            }


class AbnormalToNormalLoss(nn.Module):
    """
    Multi-component loss for training abnormal-to-normal generator
    """
    
    def __init__(self, lambda_identity=10.0, lambda_healing=2.0, lambda_adversarial=1.0, 
                 lambda_perceptual=1.0):
        super().__init__()
        self.lambda_identity = lambda_identity
        self.lambda_healing = lambda_healing
        self.lambda_adversarial = lambda_adversarial
        self.lambda_perceptual = lambda_perceptual
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for discriminator
    
    def forward(self, normal_imgs, abnormal_imgs, healed_normal, healed_abnormal, discriminator):
        """
        Compute comprehensive loss for abnormal-to-normal training
        """
        
        # 1. Identity Loss: Normal images should remain unchanged
        identity_loss = self.l1_loss(healed_normal, normal_imgs)
        
        # 2. Healing Loss: Encourage meaningful but controlled changes for abnormal images
        # We want some change but not too drastic
        healing_loss = self.l1_loss(healed_abnormal, abnormal_imgs)  # Minimize this slightly
        
        # 3. Adversarial Loss: Healed images should fool discriminator
        # Both healed normal and healed abnormal should look "normal" to discriminator
        valid_labels = torch.ones(healed_normal.size(0), 1, device=healed_normal.device)
        
        adv_loss_normal = self.adversarial_loss(discriminator(healed_normal), valid_labels)
        adv_loss_abnormal = self.adversarial_loss(discriminator(healed_abnormal), valid_labels)
        adversarial_loss = (adv_loss_normal + adv_loss_abnormal) / 2
        
        # 4. Perceptual Loss: Encourage realistic healing
        # Healed abnormal images should be perceptually similar to normal distribution
        perceptual_loss = self.mse_loss(healed_abnormal.mean(dim=[2,3]), 
                                       normal_imgs.mean(dim=[2,3]))
        
        # Total Generator Loss
        total_loss = (self.lambda_identity * identity_loss + 
                     self.lambda_healing * healing_loss +
                     self.lambda_adversarial * adversarial_loss +
                     self.lambda_perceptual * perceptual_loss)
        
        return {
            'total_loss': total_loss,
            'identity_loss': identity_loss,
            'healing_loss': healing_loss,
            'adversarial_loss': adversarial_loss,
            'segmentation_loss': torch.tensor(0.0, device=normal_imgs.device),  # Zero for compatibility
            'perceptual_loss': perceptual_loss
        }
