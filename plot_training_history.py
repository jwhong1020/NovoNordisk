#!/usr/bin/env python3
"""
Plot training history from saved CSV files

Usage:
python plot_training_history.py --csv_file saved_models/datasets/preprocessed_128/training_history/training_metrics_final_*.csv
python plot_training_history.py --json_file saved_models/datasets/preprocessed_128/training_history/training_history_final_*.json
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from glob import glob

def plot_from_csv(csv_file, output_dir=None):
    """Plot training metrics from CSV file"""
    
    # Read CSV data
    df = pd.read_csv(csv_file)
    
    if len(df) == 0:
        print("Error: CSV file is empty")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Training History - {os.path.basename(csv_file)}', fontsize=16)
    
    # Plot 1: Generator and Discriminator Losses
    axes[0, 0].plot(df['epoch'], df['avg_generator_loss'], label='Generator Loss', color='blue', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['avg_discriminator_loss'], label='Discriminator Loss', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator vs Discriminator Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Detailed Generator Losses
    axes[0, 1].plot(df['epoch'], df['avg_identity_loss'], label='Identity Loss', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['avg_healing_loss'], label='Healing Loss', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['avg_adversarial_loss'], label='Adversarial Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Generator Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Segmentation and Perceptual Losses
    axes[0, 2].plot(df['epoch'], df['avg_segmentation_loss'], label='Segmentation Loss', linewidth=2)
    axes[0, 2].plot(df['epoch'], df['avg_perceptual_loss'], label='Perceptual Loss', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Segmentation and Perceptual Losses')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Learning Rates
    axes[1, 0].plot(df['epoch'], df['generator_lr'], label='Generator LR', linewidth=2)
    axes[1, 0].plot(df['epoch'], df['discriminator_lr'], label='Discriminator LR', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rates')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Plot 5: Epoch Training Time
    axes[1, 1].plot(df['epoch'], df['epoch_time'], label='Epoch Time', color='green', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_title('Training Time per Epoch')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Loss Ratios and Convergence
    axes[1, 2].plot(df['epoch'], df['avg_generator_loss'] / df['avg_discriminator_loss'], 
                   label='G/D Loss Ratio', linewidth=2, color='purple')
    axes[1, 2].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Balance')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Ratio')
    axes[1, 2].set_title('Generator/Discriminator Loss Ratio')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    
    plot_filename = os.path.join(output_dir, 'training_history_plots.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Training plots saved to: {plot_filename}")
    
    plt.show()

def plot_from_json(json_file, output_dir=None):
    """Plot training metrics from JSON file"""
    
    # Read JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    epochs_data = data['epochs']
    if not epochs_data:
        print("Error: No epoch data found in JSON file")
        return
    
    # Convert to pandas DataFrame for easier plotting
    df_data = []
    for epoch_data in epochs_data:
        row = {
            'epoch': epoch_data['epoch'],
            'epoch_time': epoch_data['epoch_time'],
            'avg_generator_loss': epoch_data['avg_generator_loss'],
            'avg_discriminator_loss': epoch_data['avg_discriminator_loss'],
            'generator_lr': epoch_data['generator_lr'],
            'discriminator_lr': epoch_data['discriminator_lr'],
            'avg_identity_loss': epoch_data.get('avg_identity_loss', 0),
            'avg_healing_loss': epoch_data.get('avg_healing_loss', 0),
            'avg_adversarial_loss': epoch_data.get('avg_adversarial_loss', 0),
            'avg_segmentation_loss': epoch_data.get('avg_segmentation_loss', 0),
            'avg_perceptual_loss': epoch_data.get('avg_perceptual_loss', 0)
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Create temporary CSV for plotting
    temp_csv = json_file.replace('.json', '_temp.csv')
    df.to_csv(temp_csv, index=False)
    
    # Plot using the CSV function
    plot_from_csv(temp_csv, output_dir)
    
    # Clean up temporary file
    os.remove(temp_csv)
    
    # Print training summary
    print(f"\nTraining Summary:")
    print(f"Dataset: {data['config']['dataset_name']}")
    print(f"Total epochs: {len(epochs_data)}")
    print(f"Batch size: {data['config']['batch_size']}")
    print(f"Learning rate: {data['config']['lr']}")
    print(f"Total training time: {data.get('total_training_time', 'N/A'):.2f}s")
    
    # Show final losses
    final_epoch = epochs_data[-1]
    print(f"\nFinal Epoch ({final_epoch['epoch']}) Losses:")
    print(f"  Generator: {final_epoch['avg_generator_loss']:.6f}")
    print(f"  Discriminator: {final_epoch['avg_discriminator_loss']:.6f}")
    print(f"  Identity: {final_epoch.get('avg_identity_loss', 'N/A'):.6f}")
    print(f"  Healing: {final_epoch.get('avg_healing_loss', 'N/A'):.6f}")

def main():
    parser = argparse.ArgumentParser(description='Plot training history from CSV or JSON files')
    parser.add_argument('--csv_file', type=str, help='Path to CSV file with training metrics')
    parser.add_argument('--json_file', type=str, help='Path to JSON file with training history')
    parser.add_argument('--output_dir', type=str, help='Directory to save plots (default: same as input file)')
    parser.add_argument('--pattern', type=str, help='Glob pattern to find the latest file (e.g., "saved_models/*/training_history/training_metrics_final_*.csv")')
    
    args = parser.parse_args()
    
    if args.pattern:
        # Find files matching pattern
        files = glob(args.pattern)
        if not files:
            print(f"No files found matching pattern: {args.pattern}")
            return
        
        # Use the most recent file
        latest_file = max(files, key=os.path.getctime)
        print(f"Using latest file: {latest_file}")
        
        if latest_file.endswith('.csv'):
            plot_from_csv(latest_file, args.output_dir)
        elif latest_file.endswith('.json'):
            plot_from_json(latest_file, args.output_dir)
        else:
            print("Error: File must be .csv or .json")
            
    elif args.csv_file:
        if not os.path.exists(args.csv_file):
            print(f"Error: CSV file not found: {args.csv_file}")
            return
        plot_from_csv(args.csv_file, args.output_dir)
        
    elif args.json_file:
        if not os.path.exists(args.json_file):
            print(f"Error: JSON file not found: {args.json_file}")
            return
        plot_from_json(args.json_file, args.output_dir)
        
    else:
        print("Error: Please provide either --csv_file, --json_file, or --pattern")
        print("Examples:")
        print("  python plot_training_history.py --csv_file saved_models/datasets/preprocessed_128/training_history/training_metrics.csv")
        print("  python plot_training_history.py --pattern 'saved_models/*/training_history/training_metrics.csv'")

if __name__ == '__main__':
    main()
