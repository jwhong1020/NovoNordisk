import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import Generator, Discriminator

class AnomalyDetector(nn.Module):
    """
    Anomaly Detection model using CycleGAN architecture.
    The model learns to reconstruct normal images and fails on anomalies.
    """
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(AnomalyDetector, self).__init__()
        
        # Single generator for normal image reconstruction
        self.generator = Generator(input_nc, output_nc, n_residual_blocks)
        
        # Discriminator to ensure realistic reconstructions
        self.discriminator = Discriminator(input_nc)
        
    def forward(self, x):
        """Forward pass returns reconstructed image"""
        return self.generator(x)
    
    def compute_concentration_penalty(self, difference_map, threshold_percentile=90):
        """
        Compute concentration penalty that heavily penalizes large patches of concentrated differences.
        
        Args:
            difference_map: Pixel-wise difference map [batch, channels, height, width]
            threshold_percentile: Percentile threshold for defining "high difference" pixels
        
        Returns:
            concentration_score: Higher values for more concentrated anomalies
        """
        batch_size = difference_map.shape[0]
        concentration_scores = []
        
        for i in range(batch_size):
            diff_map = difference_map[i, 0]  # Assume single channel
            
            # Create binary mask of high-difference pixels
            threshold = torch.quantile(diff_map.flatten(), threshold_percentile / 100.0)
            binary_mask = (diff_map > threshold).float()
            
            # Spatial concentration using distance from center of mass
            if binary_mask.sum() > 0:
                # Find center of mass of anomalous pixels
                y_coords, x_coords = torch.meshgrid(torch.arange(diff_map.shape[0], device=diff_map.device),
                                                   torch.arange(diff_map.shape[1], device=diff_map.device), indexing='ij')
                
                total_mass = binary_mask.sum()
                if total_mass > 0:
                    center_y = (y_coords * binary_mask).sum() / total_mass
                    center_x = (x_coords * binary_mask).sum() / total_mass
                    
                    # Compute average distance from center of mass
                    distances = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
                    avg_distance = (distances * binary_mask).sum() / total_mass
                    
                    # Concentration score: higher when anomalies are more clustered
                    # Normalize by image size
                    max_distance = torch.sqrt(torch.tensor(diff_map.shape[0]**2 + diff_map.shape[1]**2, device=diff_map.device))
                    concentration = 1.0 - (avg_distance / max_distance)
                    
                    # Weight by the amount of anomalous pixels
                    anomaly_ratio = total_mass / (diff_map.shape[0] * diff_map.shape[1])
                    concentration_score = concentration * anomaly_ratio
                else:
                    concentration_score = torch.tensor(0.0, device=diff_map.device)
            else:
                concentration_score = torch.tensor(0.0, device=diff_map.device)
            
            concentration_scores.append(concentration_score)
        
        return torch.stack(concentration_scores)
    
    def compute_patch_concentration_penalty(self, difference_map, patch_size=8):
        """
        Alternative method: Compute concentration using patch-based analysis.
        Heavily penalizes large patches with consistently high differences.
        
        Args:
            difference_map: Pixel-wise difference map [batch, channels, height, width]
            patch_size: Size of patches to analyze
        
        Returns:
            patch_concentration_score: Higher values for concentrated patch anomalies
        """
        batch_size = difference_map.shape[0]
        
        # Use unfold to create patches
        patches = F.unfold(difference_map, kernel_size=patch_size, stride=patch_size//2)
        # patches shape: [batch, channels*patch_size*patch_size, num_patches]
        
        # Compute mean difference for each patch
        patch_means = patches.mean(dim=1)  # [batch, num_patches]
        
        # Find the maximum patch mean (most anomalous patch)
        max_patch_scores, _ = patch_means.max(dim=1)
        
        # Compute variance of patch scores (lower variance = more concentrated)
        patch_variance = patch_means.var(dim=1)
        
        # Concentration score: high max score with low variance indicates concentration
        concentration_score = max_patch_scores * torch.exp(-patch_variance)
        
        return concentration_score
    
    def compute_anomaly_score(self, original, reconstructed, method='mse', border_margin=16, 
                            use_concentration=True, concentration_weight=2.0, concentration_method='centroid'):
        """
        Compute anomaly score based on reconstruction error, excluding border regions.
        Now includes concentration penalty for clustered anomalies.
        
        Args:
            original: Original input image
            reconstructed: Reconstructed image from generator
            method: Method for computing anomaly score ('mse', 'l1', 'combined')
            border_margin: Number of pixels to exclude from borders (default: 16)
            use_concentration: Whether to apply concentration penalty
            concentration_weight: Weight for concentration penalty (higher = more penalty for concentration)
            concentration_method: 'centroid' or 'patch' - method for computing concentration
        
        Returns:
            anomaly_score: Higher values indicate more likely anomalies
            difference_map: Pixel-wise difference map
        """
        if method == 'mse':
            # Mean Squared Error between original and reconstructed
            difference_map = torch.pow(original - reconstructed, 2)
            
        elif method == 'l1':
            # L1 (Mean Absolute Error) loss
            difference_map = torch.abs(original - reconstructed)
            
        elif method == 'combined':
            # Combined MSE + perceptual loss (simplified)
            mse_loss = torch.pow(original - reconstructed, 2)
            l1_loss = torch.abs(original - reconstructed)
            difference_map = 0.7 * mse_loss + 0.3 * l1_loss
        
        # Apply border exclusion to difference map for scoring
        scoring_region = difference_map
        if border_margin > 0:
            h, w = difference_map.shape[-2], difference_map.shape[-1]
            if h > 2 * border_margin and w > 2 * border_margin:
                scoring_region = difference_map[:, :, border_margin:-border_margin, border_margin:-border_margin]
            else:
                print(f"Warning: Image size ({h}x{w}) too small for border margin {border_margin}. Using full image.")
        
        # Base anomaly score (mean reconstruction error)
        base_score = torch.mean(scoring_region, dim=[1, 2, 3])
        
        if use_concentration:
            # Compute concentration penalty
            if concentration_method == 'centroid':
                concentration_penalty = self.compute_concentration_penalty(scoring_region)
            elif concentration_method == 'patch':
                concentration_penalty = self.compute_patch_concentration_penalty(scoring_region)
            else:
                raise ValueError(f"Unknown concentration method: {concentration_method}")
            
            # Apply concentration penalty
            # Higher concentration_weight means concentrated anomalies get much higher scores
            anomaly_score = base_score * (1.0 + concentration_weight * concentration_penalty)
        else:
            anomaly_score = base_score
            
        return anomaly_score, difference_map
    
    def detect_anomalies(self, images, threshold=None, border_margin=16, method='mse', 
                        use_concentration=True, concentration_weight=2.0, concentration_method='centroid'):
        """
        Detect anomalies in a batch of images.
        
        Args:
            images: Batch of input images
            threshold: Anomaly threshold (if None, returns raw scores)
            border_margin: Number of pixels to exclude from borders when calculating anomaly score
            method: Scoring method ('mse', 'l1', 'combined')
            use_concentration: Whether to apply concentration penalty
            concentration_weight: Weight for concentration penalty
            concentration_method: 'centroid' or 'patch' - method for computing concentration
        
        Returns:
            anomaly_scores: Anomaly scores for each image
            reconstructed: Reconstructed images
            difference_maps: Pixel-wise difference maps
            predictions: Binary predictions (if threshold provided)
        """
        self.eval()
        with torch.no_grad():
            # Reconstruct images
            reconstructed = self.generator(images)
            
            # Compute anomaly scores with concentration penalty
            anomaly_scores, difference_maps = self.compute_anomaly_score(
                images, reconstructed, method=method, border_margin=border_margin,
                use_concentration=use_concentration, concentration_weight=concentration_weight,
                concentration_method=concentration_method
            )
            
            predictions = None
            if threshold is not None:
                predictions = (anomaly_scores > threshold).float()
            
            return {
                'anomaly_scores': anomaly_scores,
                'reconstructed': reconstructed,
                'difference_maps': difference_maps,
                'predictions': predictions
            }


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.
    Helps with better reconstruction quality by comparing high-level features.
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Use VGG16 features
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:16])  # Up to relu3_3
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        """Compute perceptual loss between x and y"""
        x_features = self.features(x)
        y_features = self.features(y)
        return F.mse_loss(x_features, y_features)


class AnomalyLoss(nn.Module):
    """
    Combined loss function for anomaly detection training.
    Includes reconstruction loss, adversarial loss, and identity loss.
    """
    def __init__(self, lambda_identity=5.0, lambda_cycle=10.0, use_perceptual=True):
        super(AnomalyLoss, self).__init__()
        self.lambda_identity = lambda_identity
        self.lambda_cycle = lambda_cycle
        self.use_perceptual = use_perceptual
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        if use_perceptual:
            self.criterion_perceptual = PerceptualLoss()
    
    def forward(self, real_images, fake_images, discriminator_real_output, discriminator_fake_output):
        """
        Compute combined loss for anomaly detection.
        
        Args:
            real_images: Original normal images
            fake_images: Reconstructed images
            discriminator_real_output: Discriminator output for real images
            discriminator_fake_output: Discriminator output for fake images
        """
        # Adversarial loss (generator tries to fool discriminator)
        valid = torch.ones_like(discriminator_fake_output, requires_grad=False)
        loss_GAN = self.criterion_GAN(discriminator_fake_output, valid)
        
        # Identity loss (reconstructed image should be identical to input for normal images)
        loss_identity = self.criterion_identity(fake_images, real_images) * self.lambda_identity
        
        # Cycle consistency loss (same as identity in this case)
        loss_cycle = self.criterion_cycle(fake_images, real_images) * self.lambda_cycle
        
        # Perceptual loss (optional)
        loss_perceptual = 0
        if self.use_perceptual:
            loss_perceptual = self.criterion_perceptual(fake_images, real_images)
        
        # Total generator loss
        loss_G = loss_GAN + loss_identity + loss_cycle + loss_perceptual
        
        return {
            'loss_G': loss_G,
            'loss_GAN': loss_GAN,
            'loss_identity': loss_identity,
            'loss_cycle': loss_cycle,
            'loss_perceptual': loss_perceptual
        }
