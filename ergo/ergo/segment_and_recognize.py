import cv2
import numpy as np
from ultralytics import YOLO
import os
import pickle
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from collections import defaultdict
import time

class ObjectTracker:
    def __init__(self, max_disappeared=30):
        self.objects = {}
        self.next_object_id = 1
        self.max_disappeared = max_disappeared
        
    def register(self, box, class_name, face_name=None):
        """Register a new object"""
        self.objects[self.next_object_id] = {
            'box': box,
            'class': class_name,
            'face': face_name,
            'last_seen': time.time(),
            'disappeared': 0
        }
        object_id = self.next_object_id
        self.next_object_id += 1
        return object_id

    def update(self, boxes, class_names, face_names):
        if len(boxes) == 0:
            for object_id in list(self.objects.keys()):
                self.objects[object_id]['disappeared'] += 1
                if self.objects[object_id]['disappeared'] > self.max_disappeared:
                    del self.objects[object_id]
            return {}

        current_objects = {}
        
        for box, class_name, face_name in zip(boxes, class_names, face_names):
            best_iou = 0
            best_id = None
            
            for object_id, obj_info in self.objects.items():
                iou = self.calculate_iou(box, obj_info['box'])
                if iou > best_iou and iou > 0.3:  # IOU threshold
                    best_iou = iou
                    best_id = object_id

            if best_id is not None:
                self.objects[best_id].update({
                    'box': box,
                    'class': class_name,
                    'last_seen': time.time(),
                    'disappeared': 0
                })
                if face_name:
                    self.objects[best_id]['face'] = face_name
                current_objects[best_id] = self.objects[best_id]
            else:
                new_id = self.register(box, class_name, face_name)
                current_objects[new_id] = self.objects[new_id]

        for object_id in list(self.objects.keys()):
            if object_id not in current_objects:
                self.objects[object_id]['disappeared'] += 1
                if self.objects[object_id]['disappeared'] > self.max_disappeared:
                    del self.objects[object_id]
                else:
                    current_objects[object_id] = self.objects[object_id]

        return current_objects

    @staticmethod
    def calculate_iou(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

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

        # Initialize object tracker
        self.tracker = ObjectTracker(max_disappeared=30)

        # Camera subscriber
        self.current_frame = None
        self.camera_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_callback,
            10
        )
        
        # Processing timer
        self.timer = self.create_timer(0.1, self.process_frame)
        self.get_logger().info('YOLOFaceRecognizer node initialized')

    def camera_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def load_face_recognizer(self):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(os.path.join(self.script_dir, 'face_recognizer.yml'))
            
            with open(os.path.join(self.script_dir, 'label_map.pkl'), 'rb') as f:
                label_map = pickle.load(f)
            
            return recognizer, label_map
        except Exception as e:
            self.get_logger().error(f'Error loading face recognizer: {str(e)}')
            return None, {}

    def process_frame(self):
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()
        results = self.model(frame, task="segment", device='cpu')

        if len(results) == 0 or results[0].masks is None:
            return

        boxes = []
        class_names = []
        face_names = []

        for result in results[0].boxes.data.numpy():
            x1, y1, x2, y2, confidence, class_id = map(int, result[:6])
            class_name = results[0].names[class_id]
            face_name = None

            if class_id == 0:  # person class
                # Process face recognition
                person_crop = frame[y1:y2, x1:x2]
                gray_person = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray_person, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (fx, fy, fw, fh) in faces:
                    face_img = gray_person[fy:fy + fh, fx:fx + fw]
                    face_img = cv2.resize(face_img, (200, 200))

                    if self.recognizer is not None:
                        label, conf = self.recognizer.predict(face_img)
                        face_name = self.label_map.get(label, "Unknown")
                        
                        # Publish face image
                        face_msg = self.bridge.cv2_to_imgmsg(face_img, encoding="mono8")
                        self.face_image_pub.publish(face_msg)

            boxes.append((x1, y1, x2, y2))
            class_names.append(class_name)
            face_names.append(face_name)

        # Update tracker and get current objects
        tracked_objects = self.tracker.update(boxes, class_names, face_names)

        # Process and publish each tracked object
        for object_id, obj_info in tracked_objects.items():
            object_info = {
                "id": object_id,
                "object": obj_info['class'],
                "coordinates": obj_info['box'],
                "face": obj_info['face'],
                "last_seen": obj_info['last_seen']
            }
            
            # Draw on frame
            x1, y1, x2, y2 = obj_info['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{object_id} {obj_info['class']}"
            if obj_info['face']:
                label += f" ({obj_info['face']})"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Publish object info
            self.object_info_pub.publish(String(data=str(object_info)))

        # Publish annotated frame
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.segmented_image_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing annotated frame: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = YOLOFaceRecognizerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
