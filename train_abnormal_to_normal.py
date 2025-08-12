#!/usr/bin/env python3
"""
Training script for Abnormal-to-Normal Generator
Trains a model that "heals" abnormal images to look normal

Usage:
python train_abnormal_to_normal.py --dataset_name datasets/preprocessed_256 --n_epochs 30 --batch_size 8
"""

import argparse
import os
import numpy as np
import itertools
from tqdm import tqdm
import time
import json
import csv
from datetime import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn

from abnormal_to_normal_model import AbnormalToNormalDetector, AbnormalToNormalLoss
from anomaly_datasets import NormalOnlyDatasetGrayscale, AnomalyDatasetGrayscale
from utils import ReplayBuffer, LambdaLR, weights_init_normal

def sample_images(model, normal_loader, abnormal_loader, dataset_name, batches_done, device):
    """Save sample healing results"""
    model.eval()
    
    try:
        # Get samples
        normal_batch = next(iter(normal_loader))
        abnormal_batch = next(iter(abnormal_loader))
        
        normal_imgs = Variable(normal_batch['A']).to(device)
        abnormal_imgs = Variable(abnormal_batch['image']).to(device)
        
        with torch.no_grad():
            # Heal images
            healed_normal = model(normal_imgs)
            healed_abnormal = model(abnormal_imgs)
            
            # Get segmentation masks (placeholder scores for sampling)
            placeholder_scores = torch.ones(normal_imgs.size(0), device=device) * 0.15  # Above threshold
            seg_normal = model.get_segmentation_mask(normal_imgs, healed_normal, placeholder_scores)
            seg_abnormal = model.get_segmentation_mask(abnormal_imgs, healed_abnormal, placeholder_scores)
            
            # Create comparison grids
            sample_size = min(4, normal_imgs.size(0), abnormal_imgs.size(0))
            
            # Normal: Original vs Healed (should be similar)
            normal_comparison = torch.cat([
                normal_imgs[:sample_size], 
                healed_normal[:sample_size]
            ], 0)
            
            # Abnormal: Original vs Healed (should show healing)
            abnormal_comparison = torch.cat([
                abnormal_imgs[:sample_size], 
                healed_abnormal[:sample_size]
            ], 0)
            
            # Segmentation masks for abnormal images
            seg_visualization = seg_abnormal[:sample_size]
            
            # Save images
            save_image(normal_comparison, f"images/{dataset_name}/normal_healing_{batches_done}.png", 
                      nrow=sample_size, normalize=True)
            save_image(abnormal_comparison, f"images/{dataset_name}/abnormal_healing_{batches_done}.png", 
                      nrow=sample_size, normalize=True)
            save_image(seg_visualization, f"images/{dataset_name}/segmentation_{batches_done}.png", 
                      nrow=sample_size, normalize=True)
            
            print(f"Sample images saved: normal_healing_{batches_done}.png, abnormal_healing_{batches_done}.png, segmentation_{batches_done}.png")
            
    except Exception as e:
        print(f"Error saving sample images: {e}")
    
    model.train()

def save_training_metrics(epoch_data, dataset_name):
    """Append training metrics for one epoch to CSV file"""
    try:
        # Create output directory
        history_dir = f'saved_models/{dataset_name}/training_history'
        os.makedirs(history_dir, exist_ok=True)
        
        # Single CSV file for all training metrics
        csv_filename = f'{history_dir}/training_metrics.csv'
        
        # Prepare row data
        row = {
            'epoch': epoch_data['epoch'],
            'avg_generator_loss': epoch_data['avg_generator_loss'],
            'avg_discriminator_loss': epoch_data['avg_discriminator_loss'],
            'avg_identity_loss': epoch_data['avg_identity_loss'],
            'avg_healing_loss': epoch_data['avg_healing_loss'],
            'avg_adversarial_loss': epoch_data['avg_adversarial_loss'],
            'avg_segmentation_loss': epoch_data['avg_segmentation_loss'],
            'avg_perceptual_loss': epoch_data['avg_perceptual_loss'],
            'epoch_time': epoch_data['epoch_time'],
            'generator_lr': epoch_data['generator_lr'],
            'discriminator_lr': epoch_data['discriminator_lr']
        }
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(csv_filename)
        
        # Append to CSV
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
        return csv_filename
        
    except Exception as e:
        print(f"Warning: Could not save training metrics: {e}")
        return None

def create_balanced_dataloaders(normal_dataset, abnormal_dataset, batch_size, n_cpu):
    """Create balanced dataloaders that have the same number of batches"""
    
    # Calculate the minimum number of samples to balance the datasets
    min_size = min(len(normal_dataset), len(abnormal_dataset))
    
    # Create subset indices
    normal_indices = torch.randperm(len(normal_dataset))[:min_size]
    abnormal_indices = torch.randperm(len(abnormal_dataset))[:min_size]
    
    # Create subset datasets
    normal_subset = torch.utils.data.Subset(normal_dataset, normal_indices)
    abnormal_subset = torch.utils.data.Subset(abnormal_dataset, abnormal_indices)
    
    # Create dataloaders
    normal_dataloader = DataLoader(normal_subset, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    abnormal_dataloader = DataLoader(abnormal_subset, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    
    print(f"Balanced datasets: {len(normal_subset)} normal, {len(abnormal_subset)} abnormal images")
    
    return normal_dataloader, abnormal_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--dataset_name', type=str, default='datasets/preprocessed_256', help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
    parser.add_argument('--decay_epoch', type=int, default=2, help='epoch from which to start lr decay')
    parser.add_argument('--img_height', type=int, default=-1, help='size of image height (-1 to preserve original)')
    parser.add_argument('--img_width', type=int, default=-1, help='size of image width (-1 to preserve original)')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=100, help='interval between saving sample images')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model checkpoints')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--save_losses', action='store_true', help='save training losses and metrics to files')
    
    # Loss weights
    parser.add_argument('--lambda_identity', type=float, default=10.0, help='identity loss weight')
    parser.add_argument('--lambda_healing', type=float, default=2.0, help='healing loss weight') 
    parser.add_argument('--lambda_adversarial', type=float, default=1.0, help='adversarial loss weight')
    parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='perceptual loss weight')
    
    opt = parser.parse_args()
    print("Abnormal-to-Normal Training Configuration:")
    print(opt)

    # Create directories
    os.makedirs(f'images/{opt.dataset_name}', exist_ok=True)
    os.makedirs(f'saved_models/{opt.dataset_name}', exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = AbnormalToNormalDetector(opt.channels, opt.channels, opt.n_residual_blocks)
    criterion = AbnormalToNormalLoss(
        lambda_identity=opt.lambda_identity,
        lambda_healing=opt.lambda_healing, 
        lambda_adversarial=opt.lambda_adversarial,
        lambda_perceptual=opt.lambda_perceptual
    )

    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Load pretrained model if resuming
    if opt.epoch != 0:
        model_path = f'saved_models/{opt.dataset_name}/abnormal_to_normal_{opt.epoch-1}.pth'
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
            print(f"Model not found at {model_path}, starting from scratch")
            model.apply(weights_init_normal)
    else:
        # Initialize weights
        model.apply(weights_init_normal)

    # Optimizers (removed segmentation_head since it no longer exists)
    optimizer_G = torch.optim.Adam(
        model.healing_generator.parameters(),
        lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    effective_decay_epoch = min(opt.decay_epoch, opt.n_epochs - 1)
    if effective_decay_epoch <= 0:
        effective_decay_epoch = max(1, opt.n_epochs // 2)
    
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, effective_decay_epoch).step
    )
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, effective_decay_epoch).step
    )

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

    # Training data loading
    print("Loading datasets...")
    
    # Normal images for identity training
    normal_dataset = NormalOnlyDatasetGrayscale(opt.dataset_name, transforms_=transforms_, mode='train')
    
    # Abnormal images for healing training - only CNV
    print("Loading CNV abnormal images...")
    disease_dir = f"{opt.dataset_name}/train/CNV"
    
    if not os.path.exists(disease_dir):
        print(f"Error: CNV directory not found at {disease_dir}")
        exit(1)
    
    try:
        cnv_dataset = AnomalyDatasetGrayscale(
            normal_dir=f"{opt.dataset_name}/train/NORMAL",
            abnormal_dir=disease_dir,
            transform=transforms.Compose(transforms_),
            balance_dataset=False,
            mode='abnormal_only'  # Only load abnormal images
        )
        print(f"Loaded {len(cnv_dataset)} CNV abnormal images for training")
    except Exception as e:
        print(f"Error: Could not load CNV dataset: {e}")
        exit(1)
    
    
    # Create balanced dataloaders - now using the CNV dataset directly
    normal_dataloader, abnormal_dataloader = create_balanced_dataloaders(
        normal_dataset, cnv_dataset, opt.batch_size, opt.n_cpu
    )

    print(f"Training batches: {len(normal_dataloader)} normal, {len(abnormal_dataloader)} abnormal")

    # ----------
    #  Training
    # ----------

    print("\nStarting abnormal-to-normal training...")
    total_start_time = time.time()
    
    # Create epoch progress bar
    epoch_pbar = tqdm(range(opt.epoch, opt.n_epochs), desc="Training Epochs", position=0)
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        
        # Initialize detailed loss tracking for this epoch
        epoch_losses = {
            'identity_losses': [],
            'healing_losses': [],
            'adversarial_losses': [],
            'segmentation_losses': [],
            'perceptual_losses': []
        }
        
        # Create batch progress bar for this epoch
        batch_pbar = tqdm(zip(normal_dataloader, abnormal_dataloader), 
                         desc=f"Epoch {epoch}", position=1, leave=False,
                         total=min(len(normal_dataloader), len(abnormal_dataloader)))
        
        for i, (normal_batch, abnormal_batch) in enumerate(batch_pbar):
            
            # Model inputs
            normal_imgs = Variable(normal_batch['A']).to(device)
            abnormal_imgs = Variable(abnormal_batch['image']).to(device)
            
            # Ensure same batch size
            min_batch_size = min(normal_imgs.size(0), abnormal_imgs.size(0))
            normal_imgs = normal_imgs[:min_batch_size]
            abnormal_imgs = abnormal_imgs[:min_batch_size]
            
            # Adversarial ground truths
            valid = Variable(torch.ones(min_batch_size, 1, device=device), requires_grad=False)
            fake = Variable(torch.zeros(min_batch_size, 1, device=device), requires_grad=False)

            # ------------------
            #  Train Generator
            # ------------------

            optimizer_G.zero_grad()

            # Generate healed images
            healed_normal = model.healing_generator(normal_imgs)
            healed_abnormal = model.healing_generator(abnormal_imgs)
            
            # Compute generator loss
            loss_components = criterion(
                normal_imgs, abnormal_imgs, healed_normal, healed_abnormal, model.discriminator
            )
            
            loss_G = loss_components['total_loss']
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator
            # -----------------------

            optimizer_D.zero_grad()

            # Real loss (normal images should be classified as real)
            loss_real = criterion.adversarial_loss(model.discriminator(normal_imgs), valid)
            
            # Fake loss (healed images should be classified as fake during D training)
            loss_fake_normal = criterion.adversarial_loss(model.discriminator(healed_normal.detach()), fake)
            loss_fake_abnormal = criterion.adversarial_loss(model.discriminator(healed_abnormal.detach()), fake)
            
            # Total discriminator loss
            loss_D = (loss_real + loss_fake_normal + loss_fake_abnormal) / 3
            loss_D.backward()
            optimizer_D.step()

            # Accumulate losses
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            
            # Track detailed losses for history
            if opt.save_losses:
                epoch_losses['identity_losses'].append(loss_components['identity_loss'].item())
                epoch_losses['healing_losses'].append(loss_components['healing_loss'].item())
                epoch_losses['adversarial_losses'].append(loss_components['adversarial_loss'].item())
                epoch_losses['segmentation_losses'].append(loss_components['segmentation_loss'].item())
                epoch_losses['perceptual_losses'].append(loss_components['perceptual_loss'].item())

            # Update batch progress bar
            batch_pbar.set_postfix({
                'D_loss': f'{loss_D.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}',
                'Identity': f'{loss_components["identity_loss"].item():.4f}',
                'Healing': f'{loss_components["healing_loss"].item():.4f}'
            })

            batches_done = epoch * len(normal_dataloader) + i
            
            # Sample images
            if batches_done % opt.sample_interval == 0:
                sample_images(model, normal_dataloader, abnormal_dataloader, opt.dataset_name, batches_done, device)

        # Close batch progress bar
        batch_pbar.close()
        
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_loss_G = epoch_loss_G / min(len(normal_dataloader), len(abnormal_dataloader))
        avg_loss_D = epoch_loss_D / min(len(normal_dataloader), len(abnormal_dataloader))
        
        # Calculate average detailed losses for this epoch
        avg_detailed_losses = {}
        if opt.save_losses and epoch_losses['identity_losses']:
            avg_detailed_losses = {
                'avg_identity_loss': np.mean(epoch_losses['identity_losses']),
                'avg_healing_loss': np.mean(epoch_losses['healing_losses']),
                'avg_adversarial_loss': np.mean(epoch_losses['adversarial_losses']),
                'avg_segmentation_loss': np.mean(epoch_losses['segmentation_losses']),
                'avg_perceptual_loss': np.mean(epoch_losses['perceptual_losses'])
            }
        
        # Save epoch data to training metrics
        if opt.save_losses:
            epoch_data = {
                'epoch': epoch,
                'epoch_time': epoch_time,
                'avg_generator_loss': avg_loss_G,
                'avg_discriminator_loss': avg_loss_D,
                'generator_lr': lr_scheduler_G.get_last_lr()[0],
                'discriminator_lr': lr_scheduler_D.get_last_lr()[0],
                **avg_detailed_losses
            }
            csv_file = save_training_metrics(epoch_data, opt.dataset_name)
            if csv_file and epoch % 10 == 0:  # Print confirmation every 10 epochs
                tqdm.write(f"Training metrics saved to: {csv_file}")
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'Avg_G': f'{avg_loss_G:.4f}',
            'Avg_D': f'{avg_loss_D:.4f}',
            'Time': f'{epoch_time:.1f}s',
            'LR': f'{lr_scheduler_G.get_last_lr()[0]:.2e}'
        })
        
        tqdm.write(f"Epoch {epoch} Summary: Time: {epoch_time:.2f}s, "
                  f"Avg G Loss: {avg_loss_G:.6f}, Avg D Loss: {avg_loss_D:.6f}")

        # Save model checkpoints
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            model_path = f'saved_models/{opt.dataset_name}/abnormal_to_normal_{epoch}.pth'
            torch.save(model.state_dict(), model_path)
            tqdm.write(f"Model checkpoint saved: {model_path}")

    # Close epoch progress bar
    epoch_pbar.close()

    total_time = time.time() - total_start_time
    
    print(f"\nAbnormal-to-Normal training completed in {total_time:.2f}s ({total_time/3600:.2f}h)")
    
    if opt.save_losses:
        csv_file = f'saved_models/{opt.dataset_name}/training_history/training_metrics.csv'
        if os.path.exists(csv_file):
            print(f"Training metrics saved to: {csv_file}")
            print(f"Use plot_training_history.py to visualize the results:")
            print(f"  python plot_training_history.py --csv_file {csv_file}")

if __name__ == '__main__':
    main()
