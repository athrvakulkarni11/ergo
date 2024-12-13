import cv2
import numpy as np
from ultralytics import YOLO
import os
import pickle

# Load YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Load LBPH Face Recognizer and label map
def load_face_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists('face_recognizer_model.yml'):
        recognizer.read('face_recognizer_model.yml')
        print("Loaded existing face recognizer model.")
    else:
        raise FileNotFoundError("Face recognizer model not found. Train the model first.")

    with open('label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    
    return recognizer, label_map


# Detect persons using YOLOv8 segmentation and recognize faces
def segment_and_recognize():
    recognizer, label_map = load_face_recognizer()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 segmentation inference
        results = model(frame, task="segment" , device='cpu')
        masks = results[0].masks.data.numpy()  # Segmentation masks
        boxes = results[0].boxes.data.numpy()  # Bounding boxes
        
        for box, mask in zip(boxes, masks):
            x1, y1, x2, y2, confidence, class_id = map(int, box[:6])
            
            # Check if the detected class is "person" (class_id == 0 in COCO dataset)
            if class_id == 0:
                # Apply segmentation mask to extract the person
                binary_mask = mask.astype(np.uint8) * 255
                segmented_person = cv2.bitwise_and(frame, frame, mask=binary_mask)
                
                # Display the segmented image
                cv2.imshow("Segmented Person", segmented_person)

                # Crop the segmented region for further processing
                segmented_cropped = segmented_person[y1:y2, x1:x2]

                # Convert the cropped segment to grayscale for face detection
                gray_segment = cv2.cvtColor(segmented_cropped, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray_segment, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (fx, fy, fw, fh) in faces:
                    face_image = gray_segment[fy:fy + fh, fx:fx + fw]
                    face_image = cv2.resize(face_image, (200, 200))  # Resize for consistency
                    
                    # Recognize face using LBPH recognizer
                    label, confidence = recognizer.predict(face_image)
                    name = label_map.get(label, "Unknown")
                    
                    # Draw a rectangle and label on the original frame
                    cv2.rectangle(frame, (x1 + fx, y1 + fy), (x1 + fx + fw, y1 + fy + fh), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({confidence:.2f})", 
                                (x1 + fx, y1 + fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the output frame
        cv2.imshow("YOLOv8-Segmentation + Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    try:
        segment_and_recognize()
    except FileNotFoundError as e:
        print(e)
