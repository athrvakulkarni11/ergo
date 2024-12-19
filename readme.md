# ERGO - Enhanced Robot Guidance and Object-tracking

## Overview
ERGO is a ROS2-based robot control system that combines voice commands with computer vision to enable natural interaction with robots. The system can track and follow specific people or objects using voice commands, making human-robot interaction more intuitive and efficient.

## Key Features
- **Voice Control**: Natural language voice commands for robot control
- **Person Recognition**: Face recognition for tracking specific individuals
- **Object Detection**: YOLOv8-based object detection and tracking
- **Smooth Motion**: PID-controlled movement for stable tracking
- **Multi-target Support**: Can track both people and objects
- **Real-time Processing**: Live video processing and response

## Prerequisites
- ROS2 Humble
- Python 3.10+
- GROQ API key for voice processing
- Camera (compatible with ROS2)

### Required Python Packages
```bash
pip install ultralytics opencv-python sounddevice soundfile groq python-dotenv rclpy
```

## Quick Start

1. **Environment Setup**
```bash
# Create .env file in project root
GROQ_API_KEY=your_groq_api_key_here
```

2. **Build the Package**
```bash
cd ~/ros2_ws
colcon build --packages-select ergo
source install/setup.bash
```

3. **Run the System**
```bash
# Terminal 1: Start the main voice command interface
ros2 run ergo main

# Terminal 2: Start object detection and recognition
ros2 run ergo segment_and_recognize

# Terminal 3: Start tracking node
ros2 run ergo track
```

## Voice Commands
The system understands various natural language commands:
- "Start segmentation and recognition for [person_name]"
- "Track person called [person_name]"
- "Follow [person_name]"
- "Track [object_name]"
- "Stop"

## System Architecture

### Nodes
1. **Main Node** (`main.py`)
   - Handles voice input and command processing
   - Uses GROQ API for speech-to-text
   - Manages command execution

2. **Segmentation Node** (`segment_and_recognize.py`)
   - YOLOv8 integration
   - Face recognition
   - Object detection and segmentation
   - Publishes:
     - `/detected_objects`
     - `/segmented_image`
     - `/face_image`

3. **Tracking Node** (`track.py`)
   - PID-based motion control
   - Smooth tracking behavior
   - Subscribes to `/detected_objects`
   - Publishes to `/cmd_vel`




## Author
- Athrva Kukarni (athrvakukarni11@gmail.com)

## Acknowledgments
- ROS2 Community
- Ultralytics YOLOv8
- GROQ API
```

