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
    
    def compute_anomaly_score(self, original, reconstructed, method='mse', border_margin=16):
        """
        Compute anomaly score based on reconstruction error, excluding border regions.
        
        Args:
            original: Original input image
            reconstructed: Reconstructed image from generator
            method: Method for computing anomaly score ('mse', 'ssim', 'combined')
            border_margin: Number of pixels to exclude from borders (default: 16)
        
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
        
        # Exclude border regions from anomaly score calculation
        # Extract central region excluding borders
        if border_margin > 0:
            h, w = difference_map.shape[-2], difference_map.shape[-1]
            if h > 2 * border_margin and w > 2 * border_margin:
                # Extract central region: [batch, channels, h_start:h_end, w_start:w_end]
                central_region = difference_map[:, :, border_margin:-border_margin, border_margin:-border_margin]
                anomaly_score = torch.mean(central_region, dim=[1, 2, 3])  # Average over spatial dimensions
            else:
                # If image is too small, use the full image
                print(f"Warning: Image size ({h}x{w}) too small for border margin {border_margin}. Using full image.")
                anomaly_score = torch.mean(difference_map, dim=[1, 2, 3])
        else:
            # Use full image if border_margin is 0
            anomaly_score = torch.mean(difference_map, dim=[1, 2, 3])
            
        return anomaly_score, difference_map
    
    def detect_anomalies(self, images, threshold=None, border_margin=16):
        """
        Detect anomalies in a batch of images.
        
        Args:
            images: Batch of input images
            threshold: Anomaly threshold (if None, returns raw scores)
            border_margin: Number of pixels to exclude from borders when calculating anomaly score
        
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
            
            # Compute anomaly scores (excluding border regions)
            anomaly_scores, difference_maps = self.compute_anomaly_score(images, reconstructed, border_margin=border_margin)
            
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
