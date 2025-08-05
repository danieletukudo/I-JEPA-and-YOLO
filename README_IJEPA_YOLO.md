# I-JEPA + YOLOv8 Brain Tumor Detection System

## Overview

This project implements a brain tumor detection system that combines **I-JEPA (Image Joint Embedding Predictive Architecture)** as a feature extraction backbone with **YOLOv8** for object detection. The system is designed to detect brain tumors in medical images using self-supervised learning features from I-JEPA and the efficient detection capabilities of YOLOv8.

## 🏗️ Architecture

### Core Components

1. **I-JEPA Backbone**: Self-supervised feature extractor from Facebook/Meta
2. **YOLOv8 Detection Head**: Efficient object detection framework
3. **Feature Adaptation Layers**: Transform I-JEPA features to YOLO-compatible format
4. **Multi-scale Feature Pyramid**: P3, P4, P5 feature maps for robust detection

### Model Architecture

```
Input Image (640x640)
    ↓
I-JEPA Backbone (facebook/ijepa_vith16_1k)
    ↓
Feature Extraction (Multi-scale)
    ↓
Feature Adaptation Layers
    ↓
YOLOv8 Detection Head
    ↓
Bounding Box Predictions
```

## 📁 Project Structure

```
i-jepa/
├── Brain Tumor Detector.v1i.yolov8/     # Dataset directory
│   ├── train/                          # Training images and labels
│   ├── valid/                          # Validation images and labels
│   ├── test/                           # Test images and labels
│   └── data.yaml                       # Dataset configuration
├── ijepa_yolo_optimized.py             # Main model implementation
├── ijepa_loader_optimized.py           # I-JEPA model loader
├── test_single_image.py                # Inference and testing script
├── best_ijepa_yolo_model.pth          # Trained model weights
├── requirements.txt                    # Python dependencies
└── README_IJEPA_YOLO.md               # This documentation
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Training the Model

```bash
# Train the I-JEPA + YOLOv8 model
python ijepa_yolo_optimized.py
```

This will:
- Load and initialize the I-JEPA backbone
- Create the integrated model with YOLOv8 detection head
- Train on the brain tumor dataset
- Save the best model as `best_ijepa_yolo_model.pth`

### 3. Testing on Images

```bash
# Test on a specific image
python test_single_image.py
```

The test script will:
- Load the trained model
- Process the input image
- Detect brain tumors using both YOLO predictions and image analysis
- Save detection results as `brain_tumor_detection_result.jpg`

## 🔧 Core Components Explained

### 1. I-JEPA Backbone (`ijepa_loader_optimized.py`)

**Purpose**: Efficient loading and feature extraction from I-JEPA models

**Key Features**:
- Automatic model downloading from Hugging Face
- CUDA/CPU device management
- Optimized inference with `torch.compile`
- Feature caching for performance

**Usage**:
```python
from ijepa_loader_optimized import IJEPALoaderOptimized

# Initialize loader
loader = IJEPALoaderOptimized(device='cuda')

# Extract features
features = loader.extract_features_optimized(image)
```

### 2. Integrated Model (`ijepa_yolo_optimized.py`)

**Main Classes**:

#### `IJEPABackboneOptimized`
- **Purpose**: I-JEPA backbone with feature adaptation
- **Components**:
  - I-JEPA model loading and freezing
  - Feature adaptation layers (Linear + Conv2d)
  - Multi-scale feature generation (P3, P4, P5)

#### `IJEPAYOLOv8Model`
- **Purpose**: Complete I-JEPA + YOLOv8 integration
- **Components**:
  - I-JEPA backbone for feature extraction
  - YOLOv8 detection head (`ultralytics.nn.modules.Detect`)
  - Automatic channel dimension detection

#### `BrainTumorDatasetOptimized`
- **Purpose**: Custom dataset loader for YOLO format
- **Features**:
  - Image resizing and padding
  - Label normalization
  - Consistent tensor shapes

### 3. Training Pipeline

**Key Functions**:

#### `create_optimized_training_pipeline()`
- Sets up model, data loaders, optimizer (AdamW)
- Configures learning rate scheduler (CosineAnnealingLR)
- Initializes loss function (BCEWithLogitsLoss)

#### `train_optimized_model()`
- Implements training loop with validation
- Saves best model based on validation loss
- Generates training curves visualization

### 4. Inference System (`test_single_image.py`)

**Key Functions**:

#### `test_single_image(image_path)`
- Loads trained model and processes input image
- Runs inference and post-processing
- Saves detection results

#### `process_predictions(predictions, ...)`
- Decodes YOLO raw outputs into bounding boxes
- Implements confidence thresholding
- Applies Non-Maximum Suppression (NMS)

#### `detect_tumors_by_image_analysis(image, h, w)`
- **Fallback detection method**
- Uses traditional computer vision techniques:
  - Thresholding with Otsu's method
  - Contour detection
  - Canny edge detection
  - Aspect ratio filtering

#### `draw_detections(image, detections)`
- Draws bounding boxes and confidence scores
- Uses colored rectangles and text backgrounds
- Adds crosses at detection centers

## 🧠 Technical Details

### Model Integration Process

1. **I-JEPA Feature Extraction**:
   ```python
   # Load pretrained I-JEPA model
   model = AutoModel.from_pretrained("facebook/ijepa_vith16_1k")
   
   # Extract features at multiple scales
   features = model(images, output_hidden_states=True)
   ```

2. **Feature Adaptation**:
   ```python
   # Transform I-JEPA features to YOLO format
   p3_features = self.p3_conv(features[0])  # 80x80
   p4_features = self.p4_conv(features[1])  # 40x40
   p5_features = self.p5_conv(features[2])  # 20x20
   ```

3. **YOLO Detection**:
   ```python
   # YOLOv8 detection head
   detections = self.detect([p3_features, p4_features, p5_features])
   ```

### Detection Pipeline

1. **YOLO Output Processing**:
   - Raw output shape: `[batch, 5, 8400]` (x, y, w, h, confidence)
   - Confidence thresholding (0.1-0.3)
   - Coordinate transformation to image space

2. **Image Analysis Fallback**:
   - Grayscale conversion and blurring
   - Adaptive thresholding
   - Contour detection and filtering
   - Edge detection as secondary method

3. **Post-processing**:
   - Non-Maximum Suppression (IoU-based)
   - Confidence score calculation
   - Bounding box visualization

## 📊 Performance Metrics

### Training Performance
- **Model Size**: ~2.4GB (I-JEPA + YOLOv8)
- **Inference Time**: ~8-10 seconds per image (CPU)
- **Memory Usage**: Optimized with gradient checkpointing

### Detection Quality
- **Primary Method**: YOLO predictions with confidence thresholding
- **Fallback Method**: Image analysis with contour detection
- **Visualization**: Bounding boxes with confidence scores

## 🔍 Usage Examples

### Training the Model
```python
# Run the complete training pipeline
python ijepa_yolo_optimized.py
```

### Testing on Custom Images
```python
# Modify the image path in test_single_image.py
image_path = "path/to/your/brain/image.jpg"
python test_single_image.py
```

### Batch Processing
```python
# For multiple images, modify test_single_image.py
for image_path in image_list:
    detections = test_single_image(image_path)
    # Process results
```

## 🛠️ Configuration

### Model Parameters
- **I-JEPA Model**: `facebook/ijepa_vith16_1k`
- **Input Size**: 640x640 pixels
- **Batch Size**: 8 (configurable)
- **Learning Rate**: 1e-4 (AdamW optimizer)

### Detection Parameters
- **Confidence Threshold**: 0.1-0.3 (adjustable)
- **NMS IoU Threshold**: 0.5
- **Minimum Detection Size**: 20x20 pixels

## 📈 Results and Outputs

### Generated Files
- `best_ijepa_yolo_model.pth`: Trained model weights
- `brain_tumor_detection_result.jpg`: Detection visualization
- `detection_comparison.png`: Before/after comparison
- `training_curves.png`: Training loss curves

### Detection Output
```
📊 Found 1 potential brain tumor regions
   Detection 1: Confidence 0.500
💾 Result saved to: brain_tumor_detection_result.jpg
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```python
   # Reduce batch size or use CPU
   device = 'cpu'  # in ijepa_loader_optimized.py
   ```

2. **No Detections Found**:
   ```python
   # Lower confidence threshold
   threshold = 0.01  # in test_single_image.py
   ```

3. **Model Loading Errors**:
   ```bash
   # Clear cache and reinstall
   pip uninstall transformers
   pip install transformers --no-cache-dir
   ```

### Performance Optimization

1. **GPU Acceleration**:
   ```python
   device = 'cuda'  # Enable GPU if available
   ```

2. **Model Compilation**:
   ```python
   # Already enabled in ijepa_loader_optimized.py
   model = torch.compile(model, mode="reduce-overhead")
   ```

3. **Batch Processing**:
   ```python
   # Process multiple images together
   batch_size = 4  # Adjust based on memory
   ```

## 🚀 Future Improvements

### Planned Enhancements
1. **Multi-class Detection**: Support for different tumor types
2. **Real-time Processing**: Optimize for video streams
3. **Ensemble Methods**: Combine multiple detection approaches
4. **Advanced Post-processing**: Implement more sophisticated NMS
5. **Web Interface**: Create a user-friendly web application

### Research Directions
1. **Attention Mechanisms**: Add attention layers to I-JEPA features
2. **Knowledge Distillation**: Transfer learning from larger models
3. **Data Augmentation**: Implement medical image-specific augmentations
4. **Explainability**: Add Grad-CAM visualizations

## 📚 References

- **I-JEPA Paper**: [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
- **YOLOv8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Brain Tumor Dataset**: Roboflow Brain Tumor Detector v1

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This system is designed for research and educational purposes. For clinical use, additional validation and regulatory compliance are required. 