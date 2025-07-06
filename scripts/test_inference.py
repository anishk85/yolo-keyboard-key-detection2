#!/usr/bin/env python3
"""
YOLO Keyboard Detection Inference Script with Camera Selection
Enhanced for testing trained keyboard detection model
"""

import cv2
import torch
from ultralytics import YOLO
import argparse
import os
import time
from datetime import datetime
import numpy as np
import glob

def find_available_cameras():
    """Find all available cameras"""
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height
                })
            cap.release()
    return available_cameras

def analyze_detections(results, model):
    """Analyze detections and categorize them"""
    boxes = results[0].boxes
    if boxes is None:
        return {'keys': [], 'interactions': [], 'all': []}
    
    # Key classes and interaction classes based on your data.yaml
    key_classes = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', 
                   'A', 'B', 'BACKSPACE', 'C', 'D', 'E', 'ENTER', 'F', 'G', 'H', 
                   'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                   'U', 'V', 'W', 'X', 'Y', 'Z']
    interaction_classes = ['FINGER', 'PEN']
    
    detections = {'keys': [], 'interactions': [], 'all': []}
    
    for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
        class_name = model.names[int(cls)]
        detection = {
            'class': class_name,
            'confidence': float(conf),
            'bbox': box.cpu().numpy()
        }
        detections['all'].append(detection)
        
        if class_name in key_classes:
            detections['keys'].append(detection)
        elif class_name in interaction_classes:
            detections['interactions'].append(detection)
    
    return detections

def test_single_image(model_path, image_path, confidence=0.5):
    """Test trained model on a single image with detailed analysis"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load model
    print(f"🔄 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Run inference
    print(f"🔄 Running inference on: {image_path}")
    results = model(image_path, conf=confidence)
    
    # Analyze results
    detections = analyze_detections(results, model)
    
    # Print detailed statistics
    print("\n📊 Detection Results:")
    print(f"Total detections: {len(detections['all'])}")
    print(f"Keys detected: {len(detections['keys'])}")
    print(f"Interactions detected: {len(detections['interactions'])}")
    
    if detections['keys']:
        print("\n🎹 Detected Keys:")
        for det in detections['keys']:
            print(f"  {det['class']} (confidence: {det['confidence']:.2f})")
    
    if detections['interactions']:
        print("\n👆 Detected Interactions:")
        for det in detections['interactions']:
            print(f"  {det['class']} (confidence: {det['confidence']:.2f})")
    
    # Save results
    for r in results:
        im_array = r.plot()
        
        # Create output directory
        output_dir = '../results/inference_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'detection_result_{timestamp}.jpg')
        cv2.imwrite(output_path, im_array)
        print(f"💾 Results saved to: {output_path}")
        
        # Also save a copy with original name for easy access
        quick_output = os.path.join(output_dir, 'latest_detection.jpg')
        cv2.imwrite(quick_output, im_array)
        print(f"💾 Quick access: {quick_output}")

def test_multiple_images(model_path, images_dir, confidence=0.5):
    """Test model on multiple images"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return
    
    print(f"🔄 Testing {len(image_files)} images...")
    
    # Load model
    model = YOLO(model_path)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'../results/batch_inference_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    summary = []
    for i, image_path in enumerate(image_files):
        print(f"\n📷 Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        results = model(image_path, conf=confidence)
        detections = analyze_detections(results, model)
        
        # Save result
        im_array = results[0].plot()
        output_filename = f"result_{i+1:03d}_{os.path.basename(image_path)}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, im_array)
        
        # Store summary
        summary.append({
            'image': os.path.basename(image_path),
            'total': len(detections['all']),
            'keys': len(detections['keys']),
            'interactions': len(detections['interactions']),
            'key_list': [d['class'] for d in detections['keys']],
            'interaction_list': [d['class'] for d in detections['interactions']]
        })
        
        print(f"  Keys: {len(detections['keys'])}, Interactions: {len(detections['interactions'])}")
    
    # Print summary
    print("\n📈 BATCH PROCESSING SUMMARY")
    print("=" * 60)
    for result in summary:
        print(f"{result['image']:30} | Keys: {result['keys']:2} | Interactions: {result['interactions']:2}")
    
    print(f"\n💾 All results saved in: {output_dir}")

def real_time_inference(model_path, confidence=0.5, camera_index=0):
    """Real-time inference with enhanced keyboard detection display"""
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_index}")
        print("Available cameras:")
        cameras = find_available_cameras()
        for cam in cameras:
            print(f"  Camera {cam['index']}: {cam['width']}x{cam['height']}")
        return
    
    # Get camera info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Using camera {camera_index}: {width}x{height} @ {fps:.1f}fps")
    print("🎮 Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'c' - Change confidence threshold")
    print("  'space' - Pause/Resume")
    
    frame_count = 0
    detection_count = 0
    paused = False
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'../results/webcam_inference_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                frame_count += 1
                
                # Run inference
                results = model(frame, conf=confidence, verbose=False)
                detections = analyze_detections(results, model)
                
                # Draw results on frame
                annotated_frame = results[0].plot()
                
                # Add enhanced info overlay
                info_y = 30
                cv2.putText(annotated_frame, f"Camera {camera_index} | Frame {frame_count}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                info_y += 30
                cv2.putText(annotated_frame, f"Confidence: {confidence:.2f}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                info_y += 30
                cv2.putText(annotated_frame, f"Keys: {len(detections['keys'])} | Interactions: {len(detections['interactions'])}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show detected keys in corner
                if detections['keys']:
                    key_text = "Keys: " + ", ".join([d['class'] for d in detections['keys'][:5]])
                    if len(detections['keys']) > 5:
                        key_text += f" +{len(detections['keys'])-5} more"
                    cv2.putText(annotated_frame, key_text, (10, height-60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show detected interactions
                if detections['interactions']:
                    interaction_text = "Interactions: " + ", ".join([d['class'] for d in detections['interactions']])
                    cv2.putText(annotated_frame, interaction_text, (10, height-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Count detections
                if len(detections['all']) > 0:
                    detection_count += 1
            
            # Display frame
            cv2.imshow('Keyboard Detection - Real-time', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'saved_frame_{frame_count}.jpg'
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, annotated_frame)
                print(f"💾 Frame saved: {filename}")
            elif key == ord('c'):
                new_conf = float(input("Enter new confidence threshold (0.0-1.0): "))
                if 0.0 <= new_conf <= 1.0:
                    confidence = new_conf
                    print(f"✅ Confidence updated to: {confidence}")
                else:
                    print("❌ Invalid confidence value")
            elif key == ord(' '):
                paused = not paused
                print(f"⏸️ {'Paused' if paused else 'Resumed'}")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"📊 Processed {frame_count} frames with {detection_count} detection events")
        print(f"💾 Results saved in: {output_dir}")

if __name__ == "__main__":
    # Default model path - update this with your actual model path
    default_model = '/home/anish/yolo_keyboard_2/results/training_runs/keyboard_detection_20250706_204417/weights/best.pt'
    
    parser = argparse.ArgumentParser(description='YOLO Keyboard Detection Inference with Enhanced Testing')
    parser.add_argument('--model', type=str, default=default_model, help='Path to trained model')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--images-dir', type=str, help='Directory containing test images')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time inference')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0, try 2 for DroidCam)')
    parser.add_argument('--list-cameras', action='store_true', help='List available cameras')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold (lowered for better detection)')
    
    args = parser.parse_args()
    
    print("🚀 YOLO Keyboard Detection Tester")
    print("=" * 40)
    
    if args.list_cameras:
        cameras = find_available_cameras()
        print("📹 Available cameras:")
        if cameras:
            for cam in cameras:
                print(f"  Camera {cam['index']}: {cam['width']}x{cam['height']}")
        else:
            print("  No cameras found!")
    elif args.webcam:
        print(f"🎥 Starting webcam inference (camera {args.camera})...")
        real_time_inference(args.model, args.conf, args.camera)
    elif args.images_dir:
        print(f"📁 Testing multiple images from: {args.images_dir}")
        test_multiple_images(args.model, args.images_dir, args.conf)
    elif args.image:
        print(f"🖼️ Testing single image: {args.image}")
        test_single_image(args.model, args.image, args.conf)
    else:
        print("ℹ️ Usage examples:")
        print(f"  Test single image: python {__file__} --image /path/to/image.jpg")
        print(f"  Test multiple images: python {__file__} --images-dir /path/to/images/")
        print(f"  Real-time webcam: python {__file__} --webcam")
        print(f"  DroidCam: python {__file__} --webcam --camera 2")
        print(f"  List cameras: python {__file__} --list-cameras")
        print(f"  Lower confidence: python {__file__} --image /path/to/image.jpg --conf 0.2")