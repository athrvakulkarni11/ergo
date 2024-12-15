import cv2
import numpy as np
from ultralytics import YOLO
import os
import pickle
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class YOLOFaceRecognizerPublisher(Node):
    def __init__(self):
        super().__init__('yolo_face_recognizer')
        self.bridge = CvBridge()

        # Publishers
        self.object_info_pub = self.create_publisher(String, 'detected_objects', 10)
        self.segmented_image_pub = self.create_publisher(Image, 'segmented_image', 10)
        self.face_image_pub = self.create_publisher(Image, 'face_image', 10)

        # Get the directory of this script
        self.script_dir = os.path.dirname(os.path.realpath(__file__))

        # YOLO Model
        self.model = YOLO(os.path.join(self.script_dir, 'yolov8n-seg.pt'))

        # Load Face Recognizer
        self.recognizer, self.label_map = self.load_face_recognizer()

        # Start video stream
        self.cap = cv2.VideoCapture(0)
        self.timer = self.create_timer(0.1, self.process_frame)

    def load_face_recognizer(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Paths for model and label map
        model_path = os.path.join(self.script_dir, 'face_recognizer_model.yml')
        label_map_path = os.path.join(self.script_dir, 'label_map.pkl')

        if os.path.exists(model_path):
            recognizer.read(model_path)
            self.get_logger().info("Loaded existing face recognizer model.")
        else:
            raise FileNotFoundError("Face recognizer model not found. Train the model first.")

        if os.path.exists(label_map_path):
            with open(label_map_path, 'rb') as f:
                label_map = pickle.load(f)
        else:
            raise FileNotFoundError("Label map file not found.")

        return recognizer, label_map

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results = self.model(frame, task="segment", device='cpu')

        # Check for detections and segmentation masks
        if len(results) == 0 or results[0].masks is None:
            print("No objects or segmentation masks detected.")
            return

        masks = results[0].masks.data.numpy()  # Segmentation masks
        boxes = results[0].boxes.data.numpy()
        class_names = results[0].names  # Get class names from model

        # Find the nearest object (based on area)
        nearest_object = None
        max_area = 0

        for box, mask in zip(boxes, masks):
            x1, y1, x2, y2, confidence, class_id = map(int, box[:6])
            area = (x2 - x1) * (y2 - y1)
            
            # Create object info dictionary
            object_info = {
                "object": class_names[class_id],
                "coordinates": (x1, y1, x2, y2),
                "area": area,
                "confidence": float(confidence),
                "face": None
            }

            if class_id == 0:  # 'person' class
                binary_mask = mask.astype(np.uint8) * 255
                segmented_person = cv2.bitwise_and(frame, frame, mask=binary_mask)

                # Publish segmented image
                segmented_msg = self.bridge.cv2_to_imgmsg(segmented_person, encoding="bgr8")
                self.segmented_image_pub.publish(segmented_msg)

                # Process face recognition
                segmented_cropped = segmented_person[y1:y2, x1:x2]
                gray_segment = cv2.cvtColor(segmented_cropped, cv2.COLOR_BGR2GRAY)

                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray_segment, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (fx, fy, fw, fh) in faces:
                    face_image = gray_segment[fy:fy + fh, fx:fx + fw]
                    face_image = cv2.resize(face_image, (200, 200))

                    label, conf = self.recognizer.predict(face_image)
                    name = self.label_map.get(label, "Unknown")
                    object_info["face"] = name

                    # Publish face recognition image
                    face_msg = self.bridge.cv2_to_imgmsg(face_image, encoding="mono8")
                    self.face_image_pub.publish(face_msg)

            # Update nearest object if this is the largest one so far
            if area > max_area:
                max_area = area
                nearest_object = object_info

            # Publish individual object information
            self.object_info_pub.publish(String(data=str(object_info)))

        # Publish nearest object information separately
        if nearest_object:
            nearest_msg = String(data=str({"nearest_object": nearest_object}))
            self.object_info_pub.publish(nearest_msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YOLOFaceRecognizerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
