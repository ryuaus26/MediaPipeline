import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
import math

# Load YOLOv8 model
model = YOLO('yolov8n.pt') 

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Video capture from webcam
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1  # Process only one hand
) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Run YOLOv8 inference
        results = model(frame)

        # Process MediaPipe hand landmarks
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_mediapipe = hands.process(image_rgb)
        
        
        
        # Draw YOLOv8 results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class ID
                label = model.names[cls]  # Class label
                print(label)
                if (conf > 0.5 and label == "bottle"):  
                    target_x = int((x1 + x2) * 0.5)
                    target_y = int((y1 + y2) * 0.5)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the dot at target_x and target_y
        if target_x != 0 and target_y != 0:
            cv2.circle(frame, (target_x, target_y), radius=5, color=(0, 0, 255), thickness=-1)  # Red dot with radius 5

        # Draw MediaPipe hand landmarks
        if results_mediapipe.multi_hand_landmarks:
            for hand_landmarks in results_mediapipe.multi_hand_landmarks:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    
                   if (i == 8):
                        x_px = int(landmark.x * frame_width)
                        y_px = int(landmark.y * frame_height)
                        distance = math.sqrt(math.pow((target_x - x_px), 2) + math.pow((target_y - y_px), 2))

                        if distance < 50:
                            print("SUCCESS")
                        #print(f'Landmark {i}: (x: {x_px}, y: {y_px}, z: {landmark.z})')
                        #print(f"TARGET X: {target_x}, TARGET Y: {target_y}")
                    
                
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
        
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('YOLOv8 and MediaPipe', cv2.flip(frame, 1))

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()
