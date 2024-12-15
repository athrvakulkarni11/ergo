import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import ast
import time


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

        # Add new parameters
        self.declare_parameter(
            'tracking_mode',
            'person',  # Can be 'person' or 'object'
            descriptor=ParameterDescriptor(
                description="Tracking mode: 'person' or 'object'"
            )
        )
        self.declare_parameter(
            'target_object',
            '',  # Empty string means track nearest object
            descriptor=ParameterDescriptor(
                description="Specific object class to track (empty for nearest)"
            )
        )
        
        # Get parameters
        self.tracking_mode = self.get_parameter('tracking_mode').value
        self.target_object = self.get_parameter('target_object').value
        self.target_person = self.get_parameter('person_to_track').value

        # Subscriber for detected_objects
        self.subscription = self.create_subscription(
            String,
            'detected_objects',
            self.object_callback,
            10
        )

        # Publisher for cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # PID Controllers
        self.linear_pid = PIDController(kp=0.5, ki=0.01, kd=0.1)
        self.angular_pid = PIDController(kp=0.5, ki=0.01, kd=0.1)

        # Parameters
        self.target_distance = 1.0  # Desired stopping distance
        self.max_linear_speed = 0.3
        self.max_angular_speed = 1.0
        self.frame_center = (320, 240)  # Assuming a 640x480 frame

        self.get_logger().info(f"Tracking initialized for: {self.target_person}")

    def object_callback(self, msg):
        try:
            # Parse object info
            data = ast.literal_eval(msg.data)
            
            # Check if this is a nearest object message
            if "nearest_object" in data:
                nearest = data["nearest_object"]
                if self.tracking_mode == 'object':
                    if not self.target_object or nearest["object"] == self.target_object:
                        self.track_object(nearest)
                    return

            # Regular object tracking logic
            object_info = data
            obj_class = object_info.get("object", "Unknown")
            person_name = object_info.get("face", "Unknown")
            coords = object_info.get("coordinates", None)

            if self.tracking_mode == 'person':
                # Person tracking mode
                if obj_class == "person":
                    if self.target_person == "Unknown" or person_name == self.target_person:
                        if coords:
                            self.track_object(object_info)
            else:
                # Object tracking mode
                if self.target_object and obj_class == self.target_object:
                    if coords:
                        self.track_object(object_info)

        except Exception as e:
            self.get_logger().error(f"Failed to process object data: {str(e)}")

    def track_object(self, object_info):
        """
        Generic tracking function for both persons and objects
        """
        coords = object_info.get("coordinates")
        if not coords:
            self.stop_robot()
            return

        x1, y1, x2, y2 = coords
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        self.follow_target(cx, cy)

    def follow_target(self, cx, cy):
        """
        Renamed from follow_person to follow_target for generic tracking
        """
        # Previous follow_person implementation remains the same
        bbox_area = abs((cx - self.frame_center[0]) * (cy - self.frame_center[1]))
        distance_to_target = max(0.1, 1000 / bbox_area)

        linear_error = distance_to_target - self.target_distance
        linear_speed = self.linear_pid.compute(0, linear_error)
        linear_speed = max(-self.max_linear_speed, min(self.max_linear_speed, linear_speed))

        angular_error = cx - self.frame_center[0]
        angular_speed = self.angular_pid.compute(0, angular_error)
        angular_speed = max(-self.max_angular_speed, min(self.max_angular_speed, angular_speed))

        if abs(linear_error) < 0.1:
            linear_speed = 0.0

        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        self.cmd_vel_pub.publish(twist)

        target_type = "person" if self.tracking_mode == "person" else "object"
        target_name = self.target_person if self.tracking_mode == "person" else self.target_object
        
        self.get_logger().info(
            f"Tracking {target_type} ({target_name}): "
            f"Linear Speed: {linear_speed:.2f}, Angular Speed: {angular_speed:.2f}"
        )

    def stop_robot(self):
        """
        Publish zero velocities to stop the robot.
        """
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    def set_target_person(self):
        """
        Update the target person dynamically based on the parameter change.
        """
        self.target_person = self.get_parameter('person_to_track').value
        self.get_logger().info(f"Tracking person updated to: {self.target_person}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectFollowerWithDynamicReconfig()

    # Callback to dynamically update the target person
    node.add_on_set_parameters_callback(lambda params: node.set_target_person())

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Object Follower with Dynamic Reconfig.')
    finally:
        node.stop_robot()  # Ensure the robot stops on shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
