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

        for box, mask in zip(boxes, masks):
            x1, y1, x2, y2, confidence, class_id = map(int, box[:6])

            if class_id == 0:  # 'person' class in COCO dataset
                binary_mask = mask.astype(np.uint8) * 255
                segmented_person = cv2.bitwise_and(frame, frame, mask=binary_mask)

                # Publish segmented image
                segmented_msg = self.bridge.cv2_to_imgmsg(segmented_person, encoding="bgr8")
                self.segmented_image_pub.publish(segmented_msg)

                # Crop for further processing
                segmented_cropped = segmented_person[y1:y2, x1:x2]
                gray_segment = cv2.cvtColor(segmented_cropped, cv2.COLOR_BGR2GRAY)

                # Face detection and recognition
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray_segment, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (fx, fy, fw, fh) in faces:
                    face_image = gray_segment[fy:fy + fh, fx:fx + fw]
                    face_image = cv2.resize(face_image, (200, 200))

                    label, conf = self.recognizer.predict(face_image)
                    name = self.label_map.get(label, "Unknown")

                    # Publish face recognition image
                    face_msg = self.bridge.cv2_to_imgmsg(face_image, encoding="mono8")
                    self.face_image_pub.publish(face_msg)

                    # Publish object information
                    object_info = {
                        "object": "person",
                        "coordinates": (x1, y1, x2, y2),
                        "face": name,
                        "confidence": conf
                    }
                    self.object_info_pub.publish(String(data=str(object_info)))

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
