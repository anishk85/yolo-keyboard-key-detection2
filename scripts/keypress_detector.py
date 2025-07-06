#!/usr/bin/env python3
"""
Improved Key Press Detection with Multiple Methods and Better Logic
"""

import numpy as np
import cv2
from ultralytics import YOLO
import math
import os
from collections import defaultdict

class ImprovedKeyPressDetector:
    def __init__(self, model_path):
        """Initialize with better class detection"""
        self.model = YOLO(model_path)
        
        # Auto-detect class types from model
        self.interaction_classes = []
        self.key_classes = {}
        
        print(f"Model classes: {self.model.names}")
        
        # Identify interaction vs key classes
        for idx, name in self.model.names.items():
            name_upper = name.upper()
            if name_upper in ['FINGER', 'PEN']:
                self.interaction_classes.append(name)
            else:
                # Map class name to display name
                self.key_classes[name] = name
        
        print(f"Interaction classes: {self.interaction_classes}")
        print(f"Key classes: {len(self.key_classes)} keys")
        
        if not self.interaction_classes:
            print("⚠️  Warning: No interaction classes (FINGER/PEN) found!")
    
    def calculate_overlap_area(self, box1, box2):
        """Calculate overlap area between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        return (x2 - x1) * (y2 - y1)
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU with better handling"""
        overlap = self.calculate_overlap_area(box1, box2)
        if overlap == 0:
            return 0.0
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - overlap
        
        return overlap / union if union > 0 else 0.0
    
    def point_in_box(self, point, box):
        """Check if point is inside box"""
        x, y = point
        return box[0] <= x <= box[2] and box[1] <= y <= box[3]
    
    def get_box_center(self, box):
        """Get center point of box"""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    def is_interaction_over_key(self, interaction_box, key_box, method='multi'):
        """
        Determine if interaction is over key using multiple methods
        """
        if method == 'multi':
            # Use multiple criteria for better accuracy
            scores = []
            
            # 1. IoU method
            iou = self.calculate_iou(interaction_box, key_box)
            scores.append(iou > 0.01)  # Very low threshold
            
            # 2. Center point method
            interaction_center = self.get_box_center(interaction_box)
            center_in_key = self.point_in_box(interaction_center, key_box)
            scores.append(center_in_key)
            
            # 3. Overlap area method
            overlap_area = self.calculate_overlap_area(interaction_box, key_box)
            key_area = (key_box[2] - key_box[0]) * (key_box[3] - key_box[1])
            overlap_ratio = overlap_area / key_area if key_area > 0 else 0
            scores.append(overlap_ratio > 0.05)  # 5% overlap
            
            # 4. Distance method
            key_center = self.get_box_center(key_box)
            distance = math.sqrt((interaction_center[0] - key_center[0])**2 + 
                               (interaction_center[1] - key_center[1])**2)
            key_diagonal = math.sqrt((key_box[2] - key_box[0])**2 + 
                                   (key_box[3] - key_box[1])**2)
            distance_ok = distance < key_diagonal * 0.7
            scores.append(distance_ok)
            
            # Return True if at least 2 out of 4 methods agree
            return sum(scores) >= 2
        
        elif method == 'center':
            interaction_center = self.get_box_center(interaction_box)
            return self.point_in_box(interaction_center, key_box)
        
        elif method == 'overlap':
            return self.calculate_iou(interaction_box, key_box) > 0.01
        
        return False
    
    def detect_key_press(self, image, confidence_threshold=0.2, debug=False):
        """
        Detect key presses with better logic and debugging
        """
        # Run detection with lower confidence to catch more objects
        results = self.model(image, verbose=False)
        
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(boxes, confs, classes):
                if conf > confidence_threshold:
                    class_name = self.model.names[int(cls)]
                    detections.append({
                        'box': box,
                        'confidence': conf,
                        'class': class_name,
                        'class_id': int(cls)
                    })
        
        # Separate keys and interactions
        keys = [d for d in detections if d['class'] in self.key_classes]
        interactions = [d for d in detections if d['class'] in self.interaction_classes]
        
        if debug:
            print(f"🔍 Debug Info:")
            print(f"  Total detections: {len(detections)}")
            print(f"  Keys detected: {len(keys)}")
            print(f"  Interactions detected: {len(interactions)}")
            
            for key in keys:
                print(f"    Key: {key['class']} ({key['confidence']:.3f})")
            
            for interaction in interactions:
                print(f"    Interaction: {interaction['class']} ({interaction['confidence']:.3f})")
        
        # Find key presses using improved logic
        pressed_keys = []
        
        for interaction in interactions:
            best_key = None
            best_score = 0
            
            for key in keys:
                if self.is_interaction_over_key(interaction['box'], key['box'], method='multi'):
                    # Calculate a combined score
                    iou = self.calculate_iou(interaction['box'], key['box'])
                    distance_score = 1.0 / (1.0 + self.calculate_distance(interaction['box'], key['box']) / 100)
                    combined_score = iou * 0.5 + distance_score * 0.5
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_key = key
            
            if best_key:
                pressed_keys.append({
                    'key': self.key_classes[best_key['class']],
                    'key_class': best_key['class'],
                    'interaction_type': interaction['class'],
                    'key_box': best_key['box'],
                    'interaction_box': interaction['box'],
                    'key_confidence': best_key['confidence'],
                    'interaction_confidence': interaction['confidence'],
                    'match_score': best_score
                })
        
        return pressed_keys, keys, interactions
    
    def calculate_distance(self, box1, box2):
        """Calculate distance between box centers"""
        center1 = self.get_box_center(box1)
        center2 = self.get_box_center(box2)
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def visualize_detections(self, image, pressed_keys, all_keys, interactions):
        """Enhanced visualization with better colors and labels"""
        img = image.copy()
        
        # Draw all keys in blue
        for key in all_keys:
            box = key['box'].astype(int)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            label = f"{key['class']} ({key['confidence']:.2f})"
            cv2.putText(img, label, (box[0], box[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw interactions in green
        for interaction in interactions:
            box = interaction['box'].astype(int)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label = f"{interaction['class']} ({interaction['confidence']:.2f})"
            cv2.putText(img, label, (box[0], box[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Highlight pressed keys in red
        for i, pressed in enumerate(pressed_keys):
            box = pressed['key_box'].astype(int)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4)
            
            # Draw connection line
            key_center = self.get_box_center(pressed['key_box'])
            interaction_center = self.get_box_center(pressed['interaction_box'])
            cv2.line(img, (int(key_center[0]), int(key_center[1])), 
                    (int(interaction_center[0]), int(interaction_center[1])), 
                    (0, 255, 255), 2)
            
            # Enhanced label
            label = f"PRESSED: {pressed['key']} (score: {pressed['match_score']:.2f})"
            cv2.putText(img, label, (box[0], box[1] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add detection summary
        summary = f"Keys: {len(all_keys)}, Interactions: {len(interactions)}, Pressed: {len(pressed_keys)}"
        cv2.putText(img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return img
    
    def run_realtime_detection(self, camera_index=2, confidence=0.2):
        """Run real-time detection with better parameters"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Could not open camera {camera_index}")
            return
        
        print(f"🎥 Running real-time detection (camera {camera_index})")
        print("Press 'q' to quit, 's' to save frame, 'd' to toggle debug")
        
        debug_mode = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect key presses
            pressed_keys, all_keys, interactions = self.detect_key_press(
                frame, confidence_threshold=confidence, debug=debug_mode
            )
            
            # Visualize
            annotated = self.visualize_detections(frame, pressed_keys, all_keys, interactions)
            
            # Print pressed keys
            if pressed_keys:
                key_names = [k['key'] for k in pressed_keys]
                print(f"🔑 Pressed: {', '.join(key_names)}")
            
            cv2.imshow("Improved Key Press Detection", annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"detection_result_{len(pressed_keys)}_keys.jpg"
                cv2.imwrite(filename, annotated)
                print(f"💾 Saved: {filename}")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function with better error handling"""
    model_path = '/home/anish/yolo_keyboard_2/results/training_runs/2ndonewithkeysimages/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please check the model path or train the model first.")
        return
    
    try:
        detector = ImprovedKeyPressDetector(model_path)
        
        # Test different confidence thresholds
        print("\n🧪 Testing different confidence thresholds:")
        print("Starting with confidence=0.2 (lower = more detections)")
        
        detector.run_realtime_detection(camera_index=2, confidence=0.2)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your model is properly trained and the camera is connected.")

if __name__ == "__main__":
    main()