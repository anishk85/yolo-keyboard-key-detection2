# YOLO Keyboard Key Detection & Key Press Recognition

An advanced computer vision system that detects keyboard keys and recognizes key presses in real-time using YOLOv8. This project combines object detection with spatial analysis to identify when fingers or pens interact with specific keyboard keys.

---

## 🎥 Demo Video

[![YOLO Keyboard Key Detection Demo](https://img.youtube.com/vi/nbPWoGe8aiw/maxresdefault.jpg)](https://youtu.be/nbPWoGe8aiw)

**Watch the full demonstration:** [YOLO Keyboard Key Detection in Action](https://youtu.be/nbPWoGe8aiw)

---

## 🚀 Features

### Core Functionality
- **Real-time Keyboard Detection**: Detects all keyboard keys (A-Z, 0-9, ENTER, BACKSPACE, etc.)
- **Key Press Recognition**: Identifies when fingers or pens press specific keys
- **Multi-method Analysis**: Uses IoU, distance, overlap, and center-point methods for accurate detection
- **Live Camera Feed**: Works with webcams, DroidCam, and other camera sources

### Technical Features
- **YOLOv8 Integration**: State-of-the-art object detection
- **GPU Optimization**: Optimized for NVIDIA RTX 4060 (8GB VRAM)
- **Data Augmentation**: Intelligent augmentation for finger/pen detection
- **Roboflow Integration**: Easy dataset import and management
- **Multi-confidence Thresholds**: Adaptive confidence for different object types

### Detection Classes
- **40 Keyboard Keys**: All alphanumeric keys plus special keys (ENTER, BACKSPACE)
- **Interaction Objects**: FINGER and PEN detection
- **Spatial Analysis**: Determines key-interaction relationships

---

## 📁 Project Structure

```
yolo_keyboard_2/
├── scripts/
│   ├── train_keyboard_detector.py      # Main training script
│   ├── keypress_detector.py            # Real-time key press detection
│   ├── test_inference.py               # Testing and validation
│   ├── augment_finger_pen.py           # Data augmentation
│   ├── data_analysis.py                # Dataset analysis tools
│   └── verify_installation.py          # System verification
├── data/
│   └── keyboard_dataset/               # Dataset (excluded from repo)
│       ├── images/
│       │   ├── train/
│       │   ├── valid/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── valid/
│       │   └── test/
│       └── data.yaml
├── results/
│   └── training_runs/                  # Training outputs (excluded)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🎯 Use Cases

- **Typing Analytics**: Monitor typing patterns and key usage
- **Educational Tools**: Learn typing with visual feedback
- **Accessibility**: Assistive technology for keyboard interaction
- **Gaming**: Custom key press detection for games
- **Research**: Human-computer interaction studies

---

## 🗂️ Dataset

**Dataset is NOT included in this repository due to size constraints.**

### Getting the Dataset
1. **Download from Roboflow**: [Keyboard Detection Dataset](https://universe.roboflow.com/keyboard-key-detection/keyboard-key-detection-3n9gk/dataset/2)
2. **Export Format**: YOLOv8 format
3. **Place in**: `data/keyboard_dataset/`

### Dataset Structure
```
data/keyboard_dataset/
├── images/
│   ├── train/          # Training images
│   ├── valid/          # Validation images
│   └── test/           # Test images
├── labels/
│   ├── train/          # Training labels (YOLO format)
│   ├── valid/          # Validation labels
│   └── test/           # Test labels
└── data.yaml           # Dataset configuration
```

### Dataset Stats
- **Total Images**: 240+
- **Classes**: 40 (38 keys + FINGER + PEN)
- **Annotations**: 500+ per class
- **Format**: YOLO bounding boxes

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python**: 3.8+
- **GPU**: NVIDIA GPU with 4GB+ VRAM (RTX 4060 recommended)
- **CUDA**: 11.0+ (for GPU acceleration)
- **RAM**: 8GB+

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/yolo-keyboard-detection.git
   cd yolo-keyboard-detection
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv yolo_keyboard_env
   source yolo_keyboard_env/bin/activate  # Linux/Mac
   # or
   yolo_keyboard_env\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python3 scripts/verify_installation.py
   ```

5. **Download Dataset**
   - Get dataset from Roboflow
   - Extract to `data/keyboard_dataset/`

---

## 🏋️‍♂️ Training

### Quick Start Training
```bash
python3 scripts/train_keyboard_detector.py
```

### Training Parameters
- **Model**: YOLOv8n (nano) for faster training
- **Epochs**: 150 with early stopping
- **Batch Size**: Auto-optimized for your GPU
- **Image Size**: 640x640
- **Augmentation**: Optimized for keyboard detection

### Training Output
```
results/training_runs/keyboard_detection_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt         # Best model weights
│   └── last.pt         # Last epoch weights
├── results.png         # Training metrics
└── confusion_matrix.png
```

---

## 🔍 Inference & Testing

### Real-time Key Press Detection
```bash
python3 scripts/keypress_detector.py
```

**Controls:**
- `q` - Quit
- `s` - Save current frame
- `d` - Toggle debug mode

### Single Image Testing
```bash
python3 scripts/test_inference.py --model path/to/best.pt --image path/to/image.jpg
```

### Webcam Testing
```bash
python3 scripts/test_inference.py --model path/to/best.pt --webcam --camera 0
```

### DroidCam Support
```bash
python3 scripts/test_inference.py --model path/to/best.pt --webcam --camera 2
```

### Batch Image Testing
```bash
python3 scripts/test_inference.py --model path/to/best.pt --images-dir path/to/images/
```

### List Available Cameras
```bash
python3 scripts/test_inference.py --list-cameras
```

---

## 🎛️ Configuration

### Model Confidence Thresholds
- **Keys**: 0.5 (higher confidence for stability)
- **Interactions**: 0.2 (lower confidence to catch more fingers/pens)

### Detection Methods
- **IoU**: Intersection over Union
- **Distance**: Center-to-center distance
- **Overlap**: Area overlap ratio
- **Center Point**: Point-in-box detection

### Camera Settings
- **Default**: Camera 0 (built-in webcam)
- **DroidCam**: Usually Camera 2
- **USB Cameras**: Camera 1, 3, etc.

---

## 🛠️ Advanced Usage

### Data Augmentation
```bash
python3 scripts/augment_finger_pen.py
```

### Dataset Analysis
```bash
python3 scripts/data_analysis.py
```

### Custom Model Training
Edit `scripts/train_keyboard_detector.py` to modify:
- Model size (yolov8n/s/m/l/x)
- Training epochs
- Batch size
- Augmentation parameters

---

## 📊 Performance Metrics

### Model Performance
- **mAP50**: 0.96+ for keyboard keys
- **mAP50**: 0.72+ for finger detection
- **Inference Speed**: ~2ms per frame
- **Memory Usage**: ~4GB VRAM

### System Requirements
- **Minimum**: GTX 1060 6GB
- **Recommended**: RTX 4060 8GB+
- **Optimal**: RTX 4070+ 12GB

---

## 🔧 Troubleshooting

### Common Issues

**1. Low Detection Accuracy**
- Lower confidence threshold
- Increase training epochs
- Add more annotated data

**2. Memory Issues**
- Reduce batch size
- Use smaller model (yolov8n)
- Enable mixed precision

**3. Camera Not Found**
- List available cameras
- Try different camera indices
- Check camera permissions

**4. Installation Problems**
- Verify CUDA installation
- Check Python version
- Update pip packages

---

## 📝 Development Notes

### Git Workflow
- **Main Branch**: Stable releases
- **Dev Branch**: Active development
- **Feature Branches**: New features

### Excluded Files
- Large datasets (`data/`)
- Model weights (`results/`)
- Cache files (`__pycache__/`)

### Code Style
- Python PEP 8 compliance
- Type hints where applicable
- Comprehensive docstrings

---

## 🤝 Contributing

1. **Fork the Repository**
2. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit Changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to Branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open Pull Request**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Libraries & Frameworks
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** - Object detection framework
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[Albumentations](https://albumentations.ai/)** - Data augmentation

### Services
- **[Roboflow](https://roboflow.com/)** - Dataset management and annotation
- **[YouTube](https://youtube.com/)** - Demo video hosting

### Hardware
- **NVIDIA RTX 4060** - GPU used for development and testing

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/yolo-keyboard-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/yolo-keyboard-detection/discussions)
- **Email**: your.email@example.com

---

## 🔗 Links

- **Demo Video**: [YouTube](https://youtu.be/nbPWoGe8aiw)
- **Dataset**: [Roboflow](https://universe.roboflow.com/keyboard-key-detection/keyboard-key-detection-3n9gk/dataset/2)
- **Documentation**: [Project Wiki](https://github.com/yourusername/yolo-keyboard-detection/wiki)

---

*Built with ❤️ for the computer vision community*