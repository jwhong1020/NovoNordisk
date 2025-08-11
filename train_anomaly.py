#!/usr/bin/env python3
"""
Grayscale anomaly detection training script.
This version uses single-channel grayscale images for better efficiency and medical imaging standards.
"""
# python train_anomaly.py --dataset_name datasets/original --n_epochs 5 --checkpoint_interval 1 --batch_size 9  --img_height 256 --img_width 256
# python train_anomaly.py --dataset_name datasets/preprocessed_256 --n_epochs 5 --checkpoint_interval 1 --batch_size 9
# python train_anomaly.py --dataset_name datasets/preprocessed_128 --n_epochs 5 --checkpoint_interval 1 --batch_size 36

import argparse
import os
import numpy as np
import datetime
import time
import sys
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch

from anomaly_models import AnomalyDetector
from anomaly_datasets import NormalOnlyDatasetGrayscale
from utils import ReplayBuffer, LambdaLR, weights_init_normal

def sample_images(anomaly_detector, val_dataloader, dataset_name, batches_done, device):
    """Saves a generated sample from the test set"""
    try:
        imgs = next(iter(val_dataloader))
        real_A = Variable(imgs['A']).to(device)
        
        # Generate reconstructions
        with torch.no_grad():
            fake_B = anomaly_detector.generator(real_A)
        
        # Compute difference map
        diff_map = torch.abs(real_A - fake_B)
        
        # Arrange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        diff_map = make_grid(diff_map, nrow=5, normalize=True)
        
        # Arrange images along y-axis
        image_grid = torch.cat((real_A, fake_B, diff_map), 1)
        save_image(image_grid, f'images/{dataset_name}/gray_{batches_done}.png', normalize=False)
        print(f"Sample images saved to images/{dataset_name}/gray_{batches_done}.png")
    except Exception as e:
        print(f"Error saving sample images: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--dataset_name', type=str, default='OCT2017', help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--decay_epoch', type=int, default=2, help='epoch from which to start lr decay')
    parser.add_argument('--img_height', type=int, default=-1, help='size of image height')
    parser.add_argument('--img_width', type=int, default=-1, help='size of image width')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels (1 for grayscale)')
    parser.add_argument('--sample_interval', type=int, default=100, help='interval between saving generator outputs')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model checkpoints')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
    parser.add_argument('--lambda_cyc', type=float, default=10.0, help='cycle loss weight')
    parser.add_argument('--lambda_id', type=float, default=5.0, help='identity loss weight')
    
    opt = parser.parse_args()
    print("Grayscale Training Configuration:")
    print(opt)

    # Create sample and checkpoint directories
    os.makedirs(f'images/{opt.dataset_name}', exist_ok=True)
    os.makedirs(f'saved_models/{opt.dataset_name}', exist_ok=True)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize anomaly detector with 1 channel (grayscale)
    anomaly_detector = AnomalyDetector(opt.channels, opt.channels, opt.n_residual_blocks)
    anomaly_detector = anomaly_detector.to(device)

    if opt.epoch != 0:
        # Load pretrained models - load from the previous epoch
        previous_epoch = opt.epoch - 1
        model_path = f'saved_models/{opt.dataset_name}/anomaly_detector_gray_{previous_epoch}.pth'
        if os.path.exists(model_path):
            anomaly_detector.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded grayscale model from {model_path} (resuming from epoch {opt.epoch})")
        else:
            print(f"Model not found at {model_path}, trying to find latest checkpoint...")
            # Try to find the latest checkpoint
            latest_epoch = -1
            for epoch_num in range(opt.epoch - 1, -1, -1):
                test_path = f'saved_models/{opt.dataset_name}/anomaly_detector_gray_{epoch_num}.pth'
                if os.path.exists(test_path):
                    latest_epoch = epoch_num
                    break
            
            if latest_epoch >= 0:
                model_path = f'saved_models/{opt.dataset_name}/anomaly_detector_gray_{latest_epoch}.pth'
                anomaly_detector.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded latest checkpoint from {model_path} (resuming from epoch {opt.epoch})")
            else:
                print(f"No previous checkpoints found, starting from scratch")
                anomaly_detector.apply(weights_init_normal)
    else:
        # Initialize weights
        anomaly_detector.apply(weights_init_normal)

    # Move loss functions to device
    criterion_GAN = criterion_GAN.to(device)
    criterion_cycle = criterion_cycle.to(device)
    criterion_identity = criterion_identity.to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(anomaly_detector.generator.parameters(), 
                                   lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(anomaly_detector.discriminator.parameters(), 
                                   lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    effective_decay_epoch = min(opt.decay_epoch, opt.n_epochs - 1)
    if effective_decay_epoch <= 0:
        effective_decay_epoch = max(1, opt.n_epochs // 2)
    
    print(f"Using decay epoch: {effective_decay_epoch}")
    
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, effective_decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, effective_decay_epoch).step)

    # Buffers of previously generated samples
    fake_buffer = ReplayBuffer()

    # Modified transforms for original size preservation
    transforms_ = []

    # Only resize if dimensions are specified and different from a "preserve" flag
    if opt.img_height != -1 and opt.img_width != -1:  # Could add -1 as "preserve original"
        transforms_.append(transforms.Resize((opt.img_height, opt.img_width)))

    transforms_.extend([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Training data loader (only normal images)
    print("Loading grayscale training dataset...")
    dataloader = DataLoader(
        NormalOnlyDatasetGrayscale(opt.dataset_name, transforms_=transforms_, mode='train'),
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=0
    )

    # Test data loader (normal images for validation)
    print("Loading grayscale validation dataset...")
    val_dataloader = DataLoader(
        NormalOnlyDatasetGrayscale(opt.dataset_name, transforms_=transforms_, mode='val'),
        batch_size=min(5, opt.batch_size), 
        shuffle=True, 
        num_workers=0
    )

    print(f"Training dataset size: {len(dataloader)} batches")
    print(f"Validation dataset size: {len(val_dataloader)} batches")

    # ----------
    #  Training
    # ----------

    print("\nStarting grayscale training...")
    total_start_time = time.time()
    
    # Create epoch progress bar
    epoch_pbar = tqdm(range(opt.epoch, opt.n_epochs), desc="Training Epochs", position=0)
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        
        # Create batch progress bar for this epoch
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch}", position=1, leave=False)
        
        for i, batch in enumerate(batch_pbar):
            # Set model input (normal images as both A and B)
            real_A = Variable(batch['A']).to(device)
            real_B = Variable(batch['B']).to(device)

            # Adversarial ground truths - match discriminator output shape [batch_size, 1]
            batch_size = real_A.size(0)
            valid = Variable(torch.ones((batch_size, 1))).to(device)
            fake = Variable(torch.zeros((batch_size, 1))).to(device)

            # ------------------
            #  Train Generator
            # ------------------

            optimizer_G.zero_grad()

            # Identity loss (normal images should be reconstructed perfectly)
            loss_id = criterion_identity(anomaly_detector.generator(real_A), real_A)

            # GAN loss
            fake_B = anomaly_detector.generator(real_A)
            loss_GAN = criterion_GAN(anomaly_detector.discriminator(fake_B), valid)

            # Cycle loss (same as identity in this case)
            loss_cycle = criterion_cycle(fake_B, real_A)

            # Total loss
            loss_G = loss_GAN + (opt.lambda_id * loss_id) + (opt.lambda_cyc * loss_cycle)

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator
            # -----------------------

            optimizer_D.zero_grad()

            # Real loss
            loss_real = criterion_GAN(anomaly_detector.discriminator(real_A), valid)
            
            # Fake loss
            fake_B_ = fake_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(anomaly_detector.discriminator(fake_B_.detach()), fake)
            
            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # Accumulate losses
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()

            # Update batch progress bar with current losses
            batch_pbar.set_postfix({
                'D_loss': f'{loss_D.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}',
                'Identity': f'{loss_id.item():.4f}',
                'Cycle': f'{loss_cycle.item():.4f}'
            })

            # Print detailed progress less frequently
            if i % 500 == 0:
                tqdm.write(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
                          f"[D loss: {loss_D.item():.6f}] [G loss: {loss_G.item():.6f}] "
                          f"[Identity: {loss_id.item():.6f}] [Cycle: {loss_cycle.item():.6f}]")

            # Save sample images
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_images(anomaly_detector, val_dataloader, opt.dataset_name, batches_done, device)

        # Close batch progress bar
        batch_pbar.close()
        
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_loss_G = epoch_loss_G / len(dataloader)
        avg_loss_D = epoch_loss_D / len(dataloader)
        
        # Update epoch progress bar with summary
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
            model_path = f'saved_models/{opt.dataset_name}/anomaly_detector_gray_{epoch}.pth'
            torch.save(anomaly_detector.state_dict(), model_path)
            tqdm.write(f"Model checkpoint saved: {model_path}")

    # Close epoch progress bar
    epoch_pbar.close()

    total_time = time.time() - total_start_time
    print(f"\nGrayscale training completed in {total_time:.2f}s ({total_time/3600:.2f}h)")

if __name__ == '__main__':
    main()
