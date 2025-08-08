#!/usr/bin/env python3
"""
Helper script to resume training from the latest checkpoint.
Automatically finds the latest saved model and resumes training.
"""

import os
import glob
import argparse
import subprocess
import sys

def find_latest_checkpoint(dataset_name, model_type='gray'):
    """Find the latest checkpoint file"""
    if model_type == 'gray':
        pattern = f'saved_models/{dataset_name}/anomaly_detector_gray_*.pth'
    else:
        pattern = f'saved_models/{dataset_name}/anomaly_detector_*.pth'
    
    checkpoint_files = glob.glob(pattern)
    if not checkpoint_files:
        return None, -1
    
    # Extract epoch numbers from filenames
    epochs = []
    for file in checkpoint_files:
        try:
            if model_type == 'gray':
                epoch = int(file.split('anomaly_detector_gray_')[1].split('.pth')[0])
            else:
                epoch = int(file.split('anomaly_detector_')[1].split('.pth')[0])
            epochs.append((epoch, file))
        except:
            continue
    
    if not epochs:
        return None, -1
    
    # Find the latest epoch
    latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
    return latest_file, latest_epoch

def main():
    parser = argparse.ArgumentParser(description='Resume anomaly detection training from latest checkpoint')
    parser.add_argument('--dataset_name', type=str, default='OCT2017', help='name of the dataset')
    parser.add_argument('--n_epochs', type=int, default=50, help='total number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--model_type', type=str, choices=['rgb', 'gray'], default='gray', help='model type to resume')
    parser.add_argument('--sample_interval', type=int, default=500, help='sample interval')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='checkpoint interval')
    
    args = parser.parse_args()
    
    print(f"🔍 Looking for {args.model_type} checkpoints for dataset {args.dataset_name}...")
    
    latest_file, latest_epoch = find_latest_checkpoint(args.dataset_name, args.model_type)
    
    if latest_file is None:
        print("❌ No checkpoints found. Starting training from scratch...")
        resume_epoch = 0
    else:
        print(f"✅ Found latest checkpoint: {latest_file} (epoch {latest_epoch})")
        resume_epoch = latest_epoch + 1
        print(f"📈 Resuming training from epoch {resume_epoch}")
    
    # Build the command
    if args.model_type == 'gray':
        script_name = 'train_anomaly_grayscale.py'
    else:
        script_name = 'train_anomaly.py'
    
    command = [
        'python', script_name,
        '--epoch', str(resume_epoch),
        '--n_epochs', str(args.n_epochs),
        '--dataset_name', args.dataset_name,
        '--batch_size', str(args.batch_size),
        '--sample_interval', str(args.sample_interval),
        '--checkpoint_interval', str(args.checkpoint_interval)
    ]
    
    print(f"🚀 Running: {' '.join(command)}")
    print("-" * 60)
    
    # Execute the training command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        sys.exit(0)

if __name__ == '__main__':
    main()
