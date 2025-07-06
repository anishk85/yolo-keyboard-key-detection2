#!/usr/bin/env python3
"""
Complete installation verification for YOLO keyboard detection project
"""

import sys
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import psutil
import os

def check_system_info():
    """Check system specifications"""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")

def check_gpu_info():
    """Check GPU and CUDA information"""
    print("\n" + "=" * 60)
    print("GPU & CUDA INFORMATION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / (1024**3):.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multiprocessors: {props.multi_processor_count}")

def check_libraries():
    """Check if all required libraries are installed"""
    print("\n" + "=" * 60)
    print("LIBRARY VERSIONS")
    print("=" * 60)
    
    libraries = [
        ('torch', torch.__version__),
        ('cv2', cv2.__version__),
        ('numpy', np.__version__),
        ('PIL', Image.__version__),
    ]
    
    try:
        import matplotlib
        libraries.append(('matplotlib', matplotlib.__version__))
    except ImportError:
        print("❌ matplotlib not installed")
    
    try:
        import pandas as pd
        libraries.append(('pandas', pd.__version__))
    except ImportError:
        print("❌ pandas not installed")
    
    for lib_name, version in libraries:
        print(f"✅ {lib_name}: {version}")

def test_yolo():
    """Test YOLO installation and functionality"""
    print("\n" + "=" * 60)
    print("YOLO FUNCTIONALITY TEST")
    print("=" * 60)
    
    try:
        # Test YOLO import and model loading
        print("Loading YOLOv8 nano model...")
        model = YOLO('yolov8n.pt')  # This will download if not present
        print("✅ YOLO model loaded successfully!")
        
        # Test inference on sample image
        print("Testing inference on sample image...")
        results = model('https://ultralytics.com/images/bus.jpg', verbose=False)
        print("✅ YOLO inference test passed!")
        
        # Test GPU inference if available
        if torch.cuda.is_available():
            print("Testing GPU inference...")
            model.to('cuda')
            results = model('https://ultralytics.com/images/bus.jpg', device=0, verbose=False)
            print("✅ GPU inference test passed!")
            
        return True
        
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage with different batch sizes"""
    if not torch.cuda.is_available():
        print("Skipping memory test - no GPU available")
        return
    
    print("\n" + "=" * 60)
    print("GPU MEMORY TEST")
    print("=" * 60)
    
    model = YOLO('yolov8n.pt').to('cuda')
    
    # Test different batch sizes
    test_sizes = [4, 8, 16, 20, 24]
    max_working_batch = 0
    
    for batch_size in test_sizes:
        try:
            torch.cuda.empty_cache()
            
            # Create dummy batch
            dummy_images = [torch.randn(3, 640, 640).cuda() for _ in range(batch_size)]
            
            # Test inference
            with torch.no_grad():
                results = model(dummy_images, verbose=False)
            
            memory_used = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"✅ Batch size {batch_size}: {memory_used:.1f} GB GPU memory")
            max_working_batch = batch_size
            
        except Exception as e:
            print(f"❌ Batch size {batch_size}: Failed - {str(e)[:50]}...")
            break
    
    print(f"Recommended batch size for training: {max_working_batch}")
    torch.cuda.empty_cache()

def main():
    """Run all verification tests"""
    print("YOLO KEYBOARD DETECTION - INSTALLATION VERIFICATION")
    
    check_system_info()
    check_gpu_info()
    check_libraries()
    
    yolo_success = test_yolo()
    
    if torch.cuda.is_available():
        test_memory_usage()
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if yolo_success and torch.cuda.is_available():
        print("🎉 ALL TESTS PASSED!")
        print("Your system is ready for YOLO keyboard detection training!")
        print("\nRecommended next steps:")
        print("1. Collect keyboard images using your phone")
        print("2. Annotate images using LabelImg")
        print("3. Run training script")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
