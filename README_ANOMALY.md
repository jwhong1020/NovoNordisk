# CycleGAN for Medical Anomaly Detection

This repository has been adapted to use CycleGAN for medical anomaly detection, specifically designed for OCT (Optical Coherence Tomography) images. The approach leverages CycleGAN's ability to learn normal image patterns and detect anomalies through reconstruction errors.

## 🧠 How It Works

### Core Concept
1. **Training on Normal Images**: The model learns to reconstruct only healthy/normal medical images
2. **Identity Mapping**: During training, normal images are mapped to themselves, forcing the model to learn the "normal" distribution
3. **Anomaly Detection**: When presented with an anomalous image, the model struggles to reconstruct it properly
4. **Reconstruction Error**: The difference between the original and reconstructed image indicates anomaly presence

### Key Components
- **Generator**: Learns to reconstruct normal images perfectly
- **Discriminator**: Ensures reconstructions look realistic
- **Anomaly Scoring**: Multiple methods (MSE, L1, perceptual) to quantify reconstruction error
- **Threshold Optimization**: Automatic threshold selection using ROC analysis

## 📁 File Structure

```
├── anomaly_datasets.py      # Custom datasets for anomaly detection
├── anomaly_models.py        # Anomaly detection models and loss functions
├── anomaly_utils.py         # Utility functions for visualization and analysis
├── train_anomaly.py         # Training script for anomaly detection
├── test_anomaly.py          # Testing and evaluation script
├── models.py               # Original CycleGAN models
├── datasets.py             # Original CycleGAN datasets
├── utils.py                # Original utilities
└── README_ANOMALY.md       # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install required packages
pip install torch torchvision matplotlib seaborn scikit-learn pandas opencv-python
```

### 2. Prepare Dataset

Organize your medical dataset as follows:
```
OCT2017/
├── train/
│   └── NORMAL/           # Only normal/healthy images
├── val/
│   └── NORMAL/           # Normal validation images
└── test/
    ├── NORMAL/           # Normal test images
    ├── CNV/              # Anomaly type 1
    ├── DME/              # Anomaly type 2
    └── DRUSEN/           # Anomaly type 3
```

**Important**: Only include NORMAL images in the training set!

### 3. Train the Model

```bash
python train_anomaly.py --dataset_name OCT2017 --n_epochs 200 --batch_size 4
```

**Training Parameters**:
- `--dataset_name`: Name of your dataset folder
- `--n_epochs`: Number of training epochs (200-300 recommended)
- `--batch_size`: Batch size (adjust based on GPU memory)
- `--lambda_id`: Identity loss weight (default: 5.0)
- `--lambda_cyc`: Cycle consistency loss weight (default: 10.0)
- `--use_perceptual`: Add perceptual loss for better quality

### 4. Test and Evaluate

```bash
python test_anomaly.py --dataset_name OCT2017 --model_epoch 200
```

**Testing Parameters**:
- `--model_epoch`: Which epoch model to load
- `--score_method`: Anomaly scoring method ('mse', 'l1', 'combined')
- `--threshold`: Manual threshold (if not provided, automatically computed)

## 📊 Understanding Results

### Output Files

After testing, you'll find these files in `output/anomaly_detection/OCT2017/`:

1. **detailed_results.csv**: Per-image results with scores and predictions
2. **roc_curve.png**: ROC curve showing model performance
3. **score_distribution.png**: Distribution of anomaly scores
4. **top_anomaly_*.png**: Visualizations of highest scoring samples
5. **anomaly_report.txt**: Comprehensive performance report

### Key Metrics

- **AUC-ROC**: Area under ROC curve (higher is better, >0.8 is good)
- **Precision/Recall**: For each class (Normal/Anomaly)
- **Optimal Threshold**: Automatically computed threshold for classification
- **Score Separation**: How well normal and anomaly scores are separated

### Visual Analysis

Each visualization shows:
- **Original**: Input medical image
- **Reconstructed**: Model's reconstruction attempt
- **Difference**: Pixel-wise difference highlighting anomalies
- **Heatmap**: Color-coded anomaly intensity

## 🔧 Advanced Usage

### Custom Anomaly Scoring

```python
from anomaly_models import AnomalyDetector

# Load trained model
model = AnomalyDetector()
model.load_state_dict(torch.load('saved_models/OCT2017/anomaly_detector_200.pth'))

# Custom scoring
results = model.detect_anomalies(images, threshold=0.05)
scores = results['anomaly_scores']
```

### Batch Processing

```python
from anomaly_utils import batch_inference

# Process entire folder
results = batch_inference(
    model=model,
    image_folder='path/to/new/images',
    output_folder='output/batch_results',
    threshold=0.05
)
```

### Hyperparameter Tuning

Key parameters to adjust:

1. **Loss Weights**:
   - `lambda_id`: Identity loss (5.0-10.0)
   - `lambda_cyc`: Cycle consistency (10.0-20.0)

2. **Architecture**:
   - `n_residual_blocks`: Generator depth (6-12)
   - Image size: 256x256 or 512x512

3. **Training**:
   - Learning rate: 0.0001-0.0002
   - Batch size: 1-8 (depending on GPU)

## 🎯 Tips for Best Results

### Dataset Preparation
- **Quality over Quantity**: Ensure normal images are truly normal
- **Diversity**: Include various normal variations (lighting, positioning)
- **Preprocessing**: Consistent image preprocessing is crucial
- **Class Balance**: Include diverse anomaly types in test set

### Training Strategies
- **Longer Training**: Medical images benefit from extended training (200+ epochs)
- **Learning Rate**: Start with 0.0002, decay after 100 epochs
- **Validation**: Monitor reconstruction quality on normal validation images
- **Early Stopping**: Stop if identity loss plateaus

### Evaluation Best Practices
- **Multiple Thresholds**: Test different threshold selection methods
- **Cross-Validation**: Validate on different patient cohorts
- **Clinical Validation**: Have medical experts review high-scoring cases
- **Baseline Comparison**: Compare with traditional methods

## 🔬 Clinical Applications

### Supported Modalities
- **OCT**: Retinal scans (current implementation)
- **X-Ray**: Chest radiographs (adapt preprocessing)
- **MRI**: Brain/organ scans (modify normalization)
- **CT**: Various anatomical regions

### Adaptation for Other Modalities

1. **Modify Dataset Class**:
   ```python
   # In anomaly_datasets.py
   transforms_ = [
       transforms.Resize((512, 512)),  # Adjust size
       transforms.Grayscale(num_output_channels=3),  # For grayscale images
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))  # Adjust normalization
   ]
   ```

2. **Adjust Model Architecture**:
   ```python
   # For different input sizes or channels
   model = AnomalyDetector(input_nc=1, output_nc=1)  # Grayscale
   ```

## 📈 Performance Optimization

### GPU Memory Management
```bash
# Reduce batch size for large images
python train_anomaly.py --batch_size 1 --img_height 512 --img_width 512

# Use gradient checkpointing for deeper models
python train_anomaly.py --n_residual_blocks 12
```

### Speed Optimization
- Use smaller image sizes during development
- Implement data loading parallelization
- Use mixed precision training for faster convergence

## 🐛 Troubleshooting

### Common Issues

1. **Low AUC Score (<0.7)**:
   - Increase training epochs
   - Adjust loss weights
   - Check data quality

2. **Poor Reconstruction Quality**:
   - Add perceptual loss
   - Increase model capacity
   - Improve data preprocessing

3. **High False Positive Rate**:
   - Lower threshold
   - Include more normal variations in training
   - Use ensemble methods

4. **Memory Issues**:
   - Reduce batch size
   - Use smaller images
   - Enable gradient checkpointing

### Getting Help

For issues and questions:
1. Check the console output for error messages
2. Verify dataset structure matches expected format
3. Ensure all dependencies are installed correctly
4. Monitor GPU memory usage during training

## 📝 Citation

If you use this code for research, please cite:

```bibtex
@article{cyclegan_anomaly_detection,
  title={CycleGAN for Medical Anomaly Detection},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional anomaly scoring methods
- Support for 3D medical images
- Integration with medical imaging standards (DICOM)
- Real-time inference optimization

---

**Happy Anomaly Hunting! 🔍🏥**
