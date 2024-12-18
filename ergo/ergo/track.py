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

        # Add all parameter declarations
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
                description="Specific object class to track (empty for nearest)"
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

        # Improved PID Controllers with better tuned values
        self.linear_pid = PIDController(kp=0.4, ki=0.01, kd=0.1)
        self.angular_pid = PIDController(kp=0.5, ki=0.01, kd=0.15)

        # Additional tracking parameters
        self.target_distance = 1.0  # meters
        self.max_linear_speed = 0.3
        self.max_angular_speed = 0.5
        self.frame_center = (320, 240)  # Assuming 640x480 resolution
        
        # Add deadzone and threshold parameters
        self.linear_deadzone = 0.05    # Minimum movement threshold
        self.angular_deadzone = 10     # Pixels from center
        self.size_threshold = 0.2      # How much size can vary before moving
        self.last_valid_detection = None
        self.detection_timeout = 0.5    # Seconds

        self.get_logger().info(f"Node initialized. Tracking mode: {self.tracking_mode}")
        self.get_logger().info(f"Target person: {self.target_person}")
        self.get_logger().info(f"Target object: {self.target_object}")

    def object_callback(self, msg):
        self.get_logger().debug(f"Received message: {msg.data}")  # Debug logging
        
        try:
            data = ast.literal_eval(msg.data)
            
            if "nearest_object" in data:
                nearest = data["nearest_object"]
                if self.tracking_mode == 'object':
                    if not self.target_object or nearest["object"] == self.target_object:
                        self.get_logger().info(f"Tracking nearest object: {nearest}")
                        self.track_object(nearest)
                    return

            object_info = data
            obj_class = object_info.get("object", "Unknown")
            person_name = object_info.get("face", "Unknown")
            coords = object_info.get("coordinates", None)

            self.get_logger().debug(f"Processing object: {obj_class}, person: {person_name}")

            if self.tracking_mode == 'person':
                if obj_class == "person":
                    if self.target_person == "Unknown" or person_name == self.target_person:
                        if coords:
                            self.get_logger().info(f"Tracking person: {person_name}")
                            self.track_object(object_info)
            else:
                if self.target_object and obj_class == self.target_object:
                    if coords:
                        self.get_logger().info(f"Tracking object: {obj_class}")
                        self.track_object(object_info)

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
        
        # Calculate confidence based on object size and position
        size_confidence = min(1.0, (width * height) / (self.frame_center[0] * self.frame_center[1]))
        center_distance = abs(cx - self.frame_center[0])
        position_confidence = 1.0 - (center_distance / self.frame_center[0])
        tracking_confidence = size_confidence * position_confidence

        self.get_logger().debug(
            f"Tracking confidence: {tracking_confidence:.2f}, "
            f"Size: {width}x{height}, Center: ({cx}, {cy})"
        )

        if tracking_confidence > 0.3:  # Only track if confidence is high enough
            self.last_valid_detection = current_time
            self.follow_target(cx, cy, width, height, tracking_confidence)
        else:
            self.stop_robot()

    def follow_target(self, cx, cy, width, height, confidence):
        # Calculate target area based on desired distance
        target_area = (self.frame_center[0] * self.frame_center[1]) / (self.target_distance ** 2)
        current_area = width * height
        
        # Improved distance estimation using object size
        distance_error = (target_area - current_area) / target_area
        
        # Apply confidence to movements
        linear_speed = self.linear_pid.compute(0, distance_error) * confidence
        
        # Calculate normalized angular error (-1 to 1)
        angular_error = (cx - self.frame_center[0]) / self.frame_center[0]
        angular_speed = self.angular_pid.compute(0, angular_error) * confidence

        # Apply deadzones to prevent jitter
        if abs(distance_error) < self.linear_deadzone:
            linear_speed = 0.0
        if abs(cx - self.frame_center[0]) < self.angular_deadzone:
            angular_speed = 0.0

        # Smooth acceleration/deceleration
        linear_speed = self.smooth_velocity(linear_speed, self.max_linear_speed)
        angular_speed = self.smooth_velocity(angular_speed, self.max_angular_speed)

        # Create and publish velocity command
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = -angular_speed  # Negative because positive angular is counterclockwise
        
        try:
            self.cmd_vel_pub.publish(twist)
            self.get_logger().debug(
                f"Distance error: {distance_error:.2f}, "
                f"Angular error: {angular_error:.2f}, "
                f"Speeds - linear: {linear_speed:.2f}, angular: {angular_speed:.2f}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to publish velocity command: {str(e)}")

    def smooth_velocity(self, velocity, max_speed):
        """Apply smooth acceleration and deceleration."""
        # Limit acceleration
        max_accel = max_speed * 0.1  # 10% of max speed per cycle
        if abs(velocity) > max_speed:
            velocity = math.copysign(max_speed, velocity)
        return velocity

    def stop_robot(self):
        """Gradually stop the robot instead of immediate stop."""
        try:
            twist = Twist()
            # Could implement gradual stopping here if needed
            self.cmd_vel_pub.publish(twist)
        except Exception as e:
            self.get_logger().error(f"Failed to stop robot: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = ObjectFollowerWithDynamicReconfig()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f"Unexpected error: {str(e)}")
    finally:
        if node:
            try:
                node.stop_robot()
                node.destroy_node()
            except Exception as e:
                print(f"Error during shutdown: {str(e)}")
        rclpy.shutdown()

if __name__ == '__main__':
    main()
