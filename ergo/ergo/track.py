import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import ast
import time
import math


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.prev_time = None

    def compute(self, setpoint, current_value):
        error = setpoint - current_value
        current_time = time.time()

        if self.prev_time is None:
            self.prev_time = current_time
        dt = current_time - self.prev_time

        # PID calculation
        proportional = self.kp * error
        self.integral += error * dt
        integral = self.ki * self.integral
        derivative = self.kd * (error - self.prev_error) / dt if dt > 0 else 0

        # Update state
        self.prev_error = error
        self.prev_time = current_time

        return proportional + integral + derivative


class ObjectFollowerWithDynamicReconfig(Node):
    def __init__(self):
        super().__init__('object_follower_dynamic_reconfig')

        # Parameter declarations
        self.declare_parameter(
            'tracking_mode',
            'person',
            descriptor=ParameterDescriptor(
                description="Tracking mode: 'person' or 'object'"
            )
        )
        self.declare_parameter(
            'target_object',
            '',
            descriptor=ParameterDescriptor(
                description="Specific object class to track"
            )
        )
        self.declare_parameter(
            'person_to_track',
            'Unknown',
            descriptor=ParameterDescriptor(
                description="Name of the person to track"
            )
        )
        
        # Get parameters
        self.tracking_mode = self.get_parameter('tracking_mode').value
        self.target_object = self.get_parameter('target_object').value
        self.target_person = self.get_parameter('person_to_track').value

        # Create subscribers and publishers
        self.subscription = self.create_subscription(
            String,
            'detected_objects',
            self.object_callback,
            10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # PID Controllers
        self.linear_pid = PIDController(kp=0.4, ki=0.01, kd=0.1)
        self.angular_pid = PIDController(kp=0.5, ki=0.01, kd=0.15)

        # Tracking parameters
        self.target_distance = 1.0  # meters
        self.max_linear_speed = 0.3
        self.max_angular_speed = 0.5
        self.frame_center = (320, 240)  # Assuming 640x480 resolution
        
        # Control parameters
        self.linear_deadzone = 0.05
        self.angular_deadzone = 10
        self.size_threshold = 0.2
        
        # Tracking state
        self.tracked_id = None
        self.last_valid_detection = None
        self.detection_timeout = 0.5

        self.get_logger().info(f"Node initialized. Tracking mode: {self.tracking_mode}")
        self.get_logger().info(f"Target person: {self.target_person}")
        self.get_logger().info(f"Target object: {self.target_object}")

    def object_callback(self, msg):
        try:
            data = ast.literal_eval(msg.data)
            object_id = data.get("id")
            
            # If we're not tracking any object yet
            if self.tracked_id is None:
                if self.tracking_mode == 'person':
                    if data.get("object") == "person":
                        face_name = data.get("face", "Unknown")
                        if self.target_person == "Unknown" or face_name == self.target_person:
                            self.tracked_id = object_id
                            self.get_logger().info(f"Started tracking person ID {object_id}: {face_name}")
                            self.track_object(data)
                else:  # object tracking mode
                    if data.get("object") == self.target_object:
                        self.tracked_id = object_id
                        self.get_logger().info(f"Started tracking {self.target_object} ID {object_id}")
                        self.track_object(data)
            
            # If we're already tracking an object
            elif object_id == self.tracked_id:
                self.track_object(data)
                
        except Exception as e:
            self.get_logger().error(f"Error processing object data: {str(e)}")

    def track_object(self, object_info):
        current_time = time.time()
        coords = object_info.get("coordinates")
        
        if not coords:
            if (self.last_valid_detection and 
                current_time - self.last_valid_detection > self.detection_timeout):
                self.stop_robot()
                self.last_valid_detection = None
            return

        x1, y1, x2, y2 = coords
        width = x2 - x1
        height = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Calculate tracking confidence
        size_confidence = min(1.0, (width * height) / (self.frame_center[0] * self.frame_center[1]))
        center_distance = abs(cx - self.frame_center[0])
        position_confidence = 1.0 - (center_distance / self.frame_center[0])
        tracking_confidence = size_confidence * position_confidence

        self.get_logger().debug(
            f"Tracking ID {object_info.get('id')} - Confidence: {tracking_confidence:.2f}, "
            f"Size: {width}x{height}, Center: ({cx}, {cy})"
        )

        if tracking_confidence > 0.3:
            self.last_valid_detection = current_time
            self.follow_target(cx, cy, width, height, tracking_confidence)
        else:
            self.stop_robot()

    def follow_target(self, cx, cy, width, height, confidence):
        # Calculate target area based on desired distance
        target_area = (self.frame_center[0] * self.frame_center[1]) / (self.target_distance ** 2)
        current_area = width * height
        
        # Distance control
        distance_error = (target_area - current_area) / target_area
        linear_speed = self.linear_pid.compute(0, distance_error) * confidence
        
        # Angular control
        angular_error = (cx - self.frame_center[0]) / self.frame_center[0]
        angular_speed = self.angular_pid.compute(0, angular_error) * confidence

        # Apply deadzones
        if abs(distance_error) < self.linear_deadzone:
            linear_speed = 0.0
        if abs(cx - self.frame_center[0]) < self.angular_deadzone:
            angular_speed = 0.0

        # Smooth velocity control
        linear_speed = self.smooth_velocity(linear_speed, self.max_linear_speed)
        angular_speed = self.smooth_velocity(angular_speed, self.max_angular_speed)

        # Create and publish velocity command
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = -angular_speed
        
        try:
            self.cmd_vel_pub.publish(twist)
            self.get_logger().debug(
                f"Following ID {self.tracked_id} - "
                f"Distance error: {distance_error:.2f}, "
                f"Angular error: {angular_error:.2f}, "
                f"Speeds - linear: {linear_speed:.2f}, angular: {angular_speed:.2f}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to publish velocity command: {str(e)}")

    def smooth_velocity(self, velocity, max_speed):
        """Apply smooth acceleration and deceleration."""
        max_accel = max_speed * 0.1
        if abs(velocity) > max_speed:
            velocity = math.copysign(max_speed, velocity)
        return velocity

    def stop_robot(self):
        """Gradually stop the robot."""
        try:
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            if self.tracked_id is not None:
                self.get_logger().info(f"Lost track of ID {self.tracked_id}")
                self.tracked_id = None
        except Exception as e:
            self.get_logger().error(f"Failed to stop robot: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectFollowerWithDynamicReconfig()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
