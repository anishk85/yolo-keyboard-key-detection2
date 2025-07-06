#!/usr/bin/env python3
"""
YOLO Keyboard Key Detection Training Script
Optimized for RTX 4060 (8GB VRAM)
"""

import torch
from ultralytics import YOLO
import os
from datetime import datetime

def get_optimal_batch_size():
    """Get optimal batch size based on available GPU memory"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # RTX 4060 with 8GB can handle batch size 16-24 for 640px images
        if gpu_memory >= 7:  # RTX 4060 shows ~8GB but usable is ~7.8GB
            return 20
        elif gpu_memory >= 5:
            return 16
        else:
            return 8
    return 4  # CPU fallback

def train_keyboard_detector():
    """Train YOLO model for keyboard key detection"""
    
    # Print system info
    print("=" * 50)
    print("YOLO Keyboard Detection Training")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load pre-trained model (start with nano for faster training)
    model = YOLO('yolov8n.pt')  # Will download if not present
    
    # Get optimal batch size
    batch_size = get_optimal_batch_size()
    print(f"Using batch size: {batch_size}")
    
    # Training parameters optimized for RTX 4060 and key press detection
    results = model.train(
        data='/home/anish/yolo_keyboard_2/data/keyboard_dataset/data.yaml',
        epochs=150,                    # Good starting point
        imgsz=640,                     # Standard size, good balance
        batch=batch_size,              # Optimized for RTX 4060
        name=f'keyboard_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        project='/home/anish/yolo_keyboard_2/results/training_runs',
        patience=25,                   # Early stopping patience
        save=True,
        plots=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=8,                     # Good for modern CPUs
        
        # Optimization parameters
        optimizer='AdamW',
        lr0=0.001,                     # Initial learning rate
        lrf=0.01,                      # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Data augmentation (moderate for keyboard detection)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,                   # Small rotation for keyboards
        translate=0.1,
        scale=0.5,
        shear=0.0,                     # No shear for keyboards
        perspective=0.0001,            # Minimal perspective
        flipud=0.0,                    # No vertical flip
        fliplr=0.0,                    # No horizontal flip for text
        mosaic=1.0,
        mixup=0.1,
        
        # Memory optimization
        amp=True,                      # Automatic Mixed Precision
        fraction=0.9,                  # Fraction of GPU memory to use
    )
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"Last model saved at: {results.save_dir}/weights/last.pt")
    print("=" * 50)
    
    return results

if __name__ == "__main__":
    # Set GPU memory growth to avoid memory allocation issues
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    train_keyboard_detector()