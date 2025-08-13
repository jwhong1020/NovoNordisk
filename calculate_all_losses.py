#!/usr/bin/env python3
"""
Fast calculation of losses and learning rates for all saved models.

This optimized script samples a small subset of data for quick loss estimation.

Usage:
python calculate_all_losses_fast.py --dataset_name datasets/preprocessed_128 --output_csv model_losses_lr.csv
"""

import argparse
import os
import glob
import csv
import re
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
import torchvision.transforms as transforms

from abnormal_to_normal_model import AbnormalToNormalDetector, AbnormalToNormalLoss
from anomaly_datasets import NormalOnlyDatasetGrayscale, AnomalyDatasetGrayscale
from utils import LambdaLR

def extract_epoch_from_filename(filename):
    """Extract epoch number from model filename"""
    match = re.search(r'abnormal_to_normal_(\d+)\.pth', filename)
    return int(match.group(1)) if match else -1

def calculate_learning_rate(epoch, total_epochs, initial_lr=0.0002, decay_epoch=None):
    """Calculate learning rate based on epoch and training schedule"""
    if decay_epoch is None:
        decay_epoch = max(1, total_epochs // 2)
    
    # Use the same LambdaLR logic as in training
    lr_lambda = LambdaLR(total_epochs, 0, decay_epoch)
    return initial_lr * lr_lambda.step(epoch)

def evaluate_model_losses_fast(model, normal_loader, abnormal_loader, criterion, device, num_samples=5):
    """
    Quick evaluation of model losses using only a few samples
    
    Args:
        model: The trained model
        normal_loader: DataLoader for normal images
        abnormal_loader: DataLoader for abnormal images  
        criterion: Loss function
        device: Device to run evaluation on
        num_samples: Number of batches to sample for evaluation
    
    Returns:
        Dictionary containing all loss components
    """
    model.eval()
    
    total_losses = {
        'total_loss': 0.0,
        'identity_loss': 0.0,
        'healing_loss': 0.0,
        'adversarial_loss': 0.0,
        'segmentation_loss': 0.0,
        'perceptual_loss': 0.0,
        'discriminator_loss': 0.0
    }
    
    samples_processed = 0
    
    with torch.no_grad():
        # Get iterators
        normal_iter = iter(normal_loader)
        abnormal_iter = iter(abnormal_loader)
        
        # Process only a few samples for speed
        for sample_idx in range(num_samples):
            try:
                # Get batches
                normal_batch = next(normal_iter)
                abnormal_batch = next(abnormal_iter)
                
                # Process inputs
                normal_imgs = Variable(normal_batch['A']).to(device)
                abnormal_imgs = Variable(abnormal_batch['image']).to(device)
                
                # Take only first image from each batch for speed
                normal_imgs = normal_imgs[:1]
                abnormal_imgs = abnormal_imgs[:1]
                
                # Generate healed images
                healed_normal = model.healing_generator(normal_imgs)
                healed_abnormal = model.healing_generator(abnormal_imgs)
                
                # Compute generator losses
                loss_components = criterion(
                    normal_imgs, abnormal_imgs, healed_normal, healed_abnormal, model.discriminator
                )
                
                # Compute discriminator loss
                valid = torch.ones(1, 1, device=device)
                fake = torch.zeros(1, 1, device=device)
                
                loss_real = criterion.adversarial_loss(model.discriminator(normal_imgs), valid)
                loss_fake_normal = criterion.adversarial_loss(model.discriminator(healed_normal), fake)
                loss_fake_abnormal = criterion.adversarial_loss(model.discriminator(healed_abnormal), fake)
                
                discriminator_loss = (loss_real + loss_fake_normal + loss_fake_abnormal) / 3
                
                # Accumulate losses
                for key in total_losses.keys():
                    if key == 'discriminator_loss':
                        total_losses[key] += discriminator_loss.item()
                    else:
                        total_losses[key] += loss_components[key].item()
                
                samples_processed += 1
                
            except StopIteration:
                break
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                continue
    
    # Average the losses
    if samples_processed > 0:
        for key in total_losses.keys():
            total_losses[key] /= samples_processed
    
    return total_losses, samples_processed

def main():
    parser = argparse.ArgumentParser(description='Fast calculation of losses and LR for all saved models')
    parser.add_argument('--dataset_name', type=str, default='datasets/preprocessed_128',
                       help='Name of the dataset')
    parser.add_argument('--output_csv', type=str, default='model_losses_lr.csv',
                       help='Output CSV filename')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size for evaluation (small for speed)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to evaluate per model for speed')
    parser.add_argument('--img_height', type=int, default=128, 
                       help='Image height')
    parser.add_argument('--img_width', type=int, default=128, 
                       help='Image width')
    parser.add_argument('--subset_size', type=int, default=50,
                       help='Size of dataset subset to use (for speed)')
    
    # Training parameters for LR calculation
    parser.add_argument('--initial_lr', type=float, default=0.0002, 
                       help='Initial learning rate used during training')
    parser.add_argument('--total_epochs', type=int, default=200,
                       help='Total epochs used during training')
    parser.add_argument('--decay_epoch', type=int, default=100,
                       help='Epoch to start learning rate decay')
    
    opt = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Fast evaluation mode: {opt.num_samples} samples per model")
    
    # Find all model files
    model_pattern = f'saved_models/{opt.dataset_name}/abnormal_to_normal_*.pth'
    model_files = glob.glob(model_pattern)
    model_files.sort(key=lambda x: extract_epoch_from_filename(x))
    
    if not model_files:
        print(f"No model files found matching pattern: {model_pattern}")
        return
    
    print(f"Found {len(model_files)} model files")
    
    # Setup data transforms
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
    transform = transforms.Compose(transforms_)
    
    # Setup datasets with small subsets for speed
    normal_dataset = NormalOnlyDatasetGrayscale(
        opt.dataset_name, 
        transforms_=transforms_,
        mode='train'
    )
    
    # For abnormal images, use CNV as the primary abnormal class
    abnormal_dataset = AnomalyDatasetGrayscale(
        normal_dir=f'{opt.dataset_name}/train/NORMAL',
        abnormal_dir=f'{opt.dataset_name}/train/CNV', 
        transform=transform,
        mode='abnormal_only'
    )
    
    # Use only small subsets for speed
    normal_indices = torch.randperm(len(normal_dataset))[:opt.subset_size]
    abnormal_indices = torch.randperm(len(abnormal_dataset))[:opt.subset_size]
    
    normal_subset = Subset(normal_dataset, normal_indices)
    abnormal_subset = Subset(abnormal_dataset, abnormal_indices)
    
    print(f"Using subsets: {len(normal_subset)} normal, {len(abnormal_subset)} abnormal")
    
    # Create data loaders with minimal workers for speed
    normal_loader = DataLoader(normal_subset, batch_size=opt.batch_size, shuffle=False, 
                              num_workers=0, drop_last=False)  # num_workers=0 for speed
    abnormal_loader = DataLoader(abnormal_subset, batch_size=opt.batch_size, shuffle=False,
                                num_workers=0, drop_last=False)
    
    # Initialize model and loss (reuse the same instances for speed)
    model = AbnormalToNormalDetector(input_nc=1, output_nc=1, n_residual_blocks=9).to(device)
    criterion = AbnormalToNormalLoss()
    
    # Prepare CSV output
    csv_filename = opt.output_csv
    fieldnames = [
        'epoch', 'model_file', 'evaluation_time', 'samples_evaluated',
        'generator_lr', 'discriminator_lr',
        'total_loss', 'identity_loss', 'healing_loss', 'adversarial_loss',
        'segmentation_loss', 'perceptual_loss', 'discriminator_loss',
        'status'
    ]
    
    # Write CSV header
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    print(f"Starting fast evaluation of {len(model_files)} models...")
    print(f"Results will be saved to: {csv_filename}")
    
    # Track overall timing
    total_start = time.time()
    
    # Evaluate each model
    for model_file in tqdm(model_files, desc="Evaluating models"):
        start_time = time.time()
        epoch = extract_epoch_from_filename(model_file)
        
        try:
            # Load model (this is the main bottleneck, but unavoidable)
            checkpoint = torch.load(model_file, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint, strict=False)
            
            # Calculate learning rates
            generator_lr = calculate_learning_rate(epoch, opt.total_epochs, opt.initial_lr, opt.decay_epoch)
            discriminator_lr = generator_lr  # Typically same for both
            
            # Fast evaluation with few samples
            losses, samples_evaluated = evaluate_model_losses_fast(
                model, normal_loader, abnormal_loader, criterion, device, opt.num_samples
            )
            
            evaluation_time = time.time() - start_time
            
            # Prepare row data
            row_data = {
                'epoch': epoch,
                'model_file': os.path.basename(model_file),
                'evaluation_time': f"{evaluation_time:.2f}",
                'samples_evaluated': samples_evaluated,
                'generator_lr': f"{generator_lr:.6f}",
                'discriminator_lr': f"{discriminator_lr:.6f}",
                'status': 'success'
            }
            
            # Add loss values
            for key, value in losses.items():
                row_data[key] = f"{value:.6f}"
            
            # Write to CSV immediately for real-time progress
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row_data)
        
        except Exception as e:
            # Handle errors gracefully
            error_time = time.time() - start_time
            error_row = {
                'epoch': epoch,
                'model_file': os.path.basename(model_file),
                'evaluation_time': f"{error_time:.2f}",
                'samples_evaluated': 0,
                'generator_lr': 'N/A',
                'discriminator_lr': 'N/A',
                'total_loss': 'N/A',
                'identity_loss': 'N/A',
                'healing_loss': 'N/A',
                'adversarial_loss': 'N/A',
                'segmentation_loss': 'N/A',
                'perceptual_loss': 'N/A',
                'discriminator_loss': 'N/A',
                'status': f'error: {str(e)[:50]}'
            }
            
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(error_row)
            
            print(f"Error evaluating epoch {epoch}: {e}")
    
    total_time = time.time() - total_start
    print(f"\nFast evaluation complete! Results saved to: {csv_filename}")
    print(f"Evaluated {len(model_files)} models in {total_time:.2f} seconds")
    print(f"Average time per model: {total_time/len(model_files):.2f} seconds")

if __name__ == '__main__':
    main()
