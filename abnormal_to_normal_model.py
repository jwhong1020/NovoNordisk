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
        
        # Segmentation head for direct abnormality localization
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(input_nc + output_nc, 64, 3, 1, 1),  # Concat input + output
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.Sigmoid()  # Output segmentation mask
        )
    
    def forward(self, x):
        """Forward pass returns healed image"""
        return self.healing_generator(x)
    
    def get_segmentation_mask(self, input_img, healed_img):
        """Generate segmentation mask from input and healed images"""
        combined = torch.cat([input_img, healed_img], dim=1)
        return self.segmentation_head(combined)
    
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
            
            # Generate segmentation masks
            segmentation_masks = self.get_segmentation_mask(images, healed_images)
            
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
                 lambda_segmentation=3.0, lambda_perceptual=1.0):
        super().__init__()
        self.lambda_identity = lambda_identity
        self.lambda_healing = lambda_healing
        self.lambda_adversarial = lambda_adversarial
        self.lambda_segmentation = lambda_segmentation
        self.lambda_perceptual = lambda_perceptual
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for discriminator
    
    def forward(self, normal_imgs, abnormal_imgs, healed_normal, healed_abnormal, 
                seg_masks_normal, seg_masks_abnormal, discriminator, 
                normal_seg_targets=None, abnormal_seg_targets=None):
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
        
        # 4. Segmentation Loss: Direct supervision for abnormality localization
        segmentation_loss = torch.tensor(0.0, device=normal_imgs.device)
        if normal_seg_targets is not None and abnormal_seg_targets is not None:
            # Normal images should have no abnormal regions
            seg_loss_normal = self.bce_loss(seg_masks_normal, 
                                          torch.zeros_like(seg_masks_normal))
            # Abnormal images should have marked abnormal regions
            seg_loss_abnormal = self.bce_loss(seg_masks_abnormal, abnormal_seg_targets)
            segmentation_loss = (seg_loss_normal + seg_loss_abnormal) / 2
        else:
            # If no segmentation targets, encourage sparse segmentation for normal images
            seg_loss_normal = self.bce_loss(seg_masks_normal, 
                                          torch.zeros_like(seg_masks_normal))
            # For abnormal images, encourage some but not too much segmentation
            seg_loss_abnormal = torch.mean(seg_masks_abnormal)  # Encourage some activation
            segmentation_loss = seg_loss_normal + 0.1 * seg_loss_abnormal
        
        # 5. Perceptual Loss: Encourage realistic healing
        # Healed abnormal images should be perceptually similar to normal distribution
        perceptual_loss = self.mse_loss(healed_abnormal.mean(dim=[2,3]), 
                                       normal_imgs.mean(dim=[2,3]))
        
        # Total Generator Loss
        total_loss = (self.lambda_identity * identity_loss + 
                     self.lambda_healing * healing_loss +
                     self.lambda_adversarial * adversarial_loss +
                     self.lambda_segmentation * segmentation_loss +
                     self.lambda_perceptual * perceptual_loss)
        
        return {
            'total_loss': total_loss,
            'identity_loss': identity_loss,
            'healing_loss': healing_loss,
            'adversarial_loss': adversarial_loss,
            'segmentation_loss': segmentation_loss,
            'perceptual_loss': perceptual_loss
        }
