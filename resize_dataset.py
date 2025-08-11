#!/usr/bin/env python3
"""
Dataset resizing utility script.
Recursively processes all images in a folder structure and resizes them to specified dimensions.
Maintains the original folder structure in the output directory.
"""

import argparse
import os
import glob
from PIL import Image
from tqdm import tqdm
import shutil

def resize_image(input_path, output_path, size):
    """
    Resize a single image and save it to the output path.
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save resized image
        size (tuple): Target size (width, height)
    """
    try:
        # Open and resize image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (handles various formats)
            if img.mode != 'RGB' and img.mode != 'L':
                if img.mode == 'RGBA':
                    # Create white background for transparent images
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    img = background
                else:
                    img = img.convert('RGB')
            
            # Resize image using high-quality resampling
            resized_img = img.resize(size, Image.Resampling.LANCZOS)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save resized image
            resized_img.save(output_path, quality=95)
            
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False
    
    return True

def get_all_images(root_dir):
    """
    Get all image files recursively from a directory.
    
    Args:
        root_dir (str): Root directory to search
        
    Returns:
        list: List of image file paths
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        # Search recursively for all image files
        pattern = os.path.join(root_dir, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
        
        # Also search for uppercase extensions
        pattern = os.path.join(root_dir, '**', ext.upper())
        image_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(list(set(image_files)))  # Remove duplicates and sort

def main():
    parser = argparse.ArgumentParser(
        description='Resize all images in a dataset while maintaining folder structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resize to 256x256
  python resize_dataset.py --input_folder datasets/preprocessed --output_folder datasets/preprocessed_256 --size 256 256
  
  # Resize to 512x512
  python resize_dataset.py --input_folder datasets/original --output_folder datasets/original_512 --size 512 512
  
  # Resize to non-square dimensions
  python resize_dataset.py --input_folder datasets/preprocessed --output_folder datasets/preprocessed_custom --size 224 224
        """
    )
    
    parser.add_argument('--input_folder', type=str, required=True, 
                        help='Path to input folder containing images')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to output folder for resized images')
    parser.add_argument('--size', type=int, nargs=2, required=True, metavar=('WIDTH', 'HEIGHT'),
                        help='Target size as width height (e.g., 256 256)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite output folder if it exists')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be processed without actually doing it')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        return 1
    
    if not os.path.isdir(args.input_folder):
        print(f"Error: '{args.input_folder}' is not a directory")
        return 1
    
    target_size = tuple(args.size)  # (width, height)
    
    # Check if output folder exists
    if os.path.exists(args.output_folder):
        if not args.overwrite:
            print(f"Error: Output folder '{args.output_folder}' already exists.")
            print("Use --overwrite flag to overwrite existing folder.")
            return 1
        else:
            print(f"Warning: Overwriting existing folder '{args.output_folder}'")
            if not args.dry_run:
                shutil.rmtree(args.output_folder)
    
    # Get all image files
    print(f"Scanning for images in '{args.input_folder}'...")
    image_files = get_all_images(args.input_folder)
    
    if not image_files:
        print("No image files found in the input folder")
        return 1
    
    print(f"Found {len(image_files)} images to process")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print(f"Output folder: {args.output_folder}")
    
    if args.dry_run:
        print("\n--- DRY RUN MODE ---")
        print("Files that would be processed:")
        for i, img_path in enumerate(image_files[:10]):  # Show first 10
            rel_path = os.path.relpath(img_path, args.input_folder)
            output_path = os.path.join(args.output_folder, rel_path)
            print(f"  {rel_path} -> {output_path}")
        if len(image_files) > 10:
            print(f"  ... and {len(image_files) - 10} more files")
        print("Use without --dry_run to actually process the images")
        return 0
    
    # Process images
    print(f"\nProcessing images...")
    successful = 0
    failed = 0
    
    # Create progress bar
    pbar = tqdm(image_files, desc="Resizing images", unit="img")
    
    for img_path in pbar:
        # Calculate relative path to maintain folder structure
        rel_path = os.path.relpath(img_path, args.input_folder)
        output_path = os.path.join(args.output_folder, rel_path)
        
        # Update progress bar with current file
        pbar.set_postfix({'file': os.path.basename(img_path)})
        
        # Resize image
        if resize_image(img_path, output_path, target_size):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} images")
    if failed > 0:
        print(f"Failed to process: {failed} images")
    
    print(f"\nResized images saved to: {args.output_folder}")
    
    # Show folder structure
    print(f"\nOutput folder structure:")
    for root, dirs, files in os.walk(args.output_folder):
        level = root.replace(args.output_folder, '').count(os.sep)
        indent = ' ' * 2 * level
        rel_path = os.path.relpath(root, args.output_folder) if root != args.output_folder else '.'
        print(f"{indent}{rel_path}/")
        subindent = ' ' * 2 * (level + 1)
        # Show first few files in each directory
        shown_files = files[:3]
        for file in shown_files:
            print(f"{subindent}{file}")
        if len(files) > 3:
            print(f"{subindent}... and {len(files) - 3} more files")
    
    return 0

if __name__ == '__main__':
    exit(main())
