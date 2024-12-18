import os
import sounddevice as sd
import soundfile as sf
from groq import Groq
import subprocess
from dotenv import load_dotenv

load_dotenv()
    
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Simplified tracking configuration
TRACKING_CONFIG = {
    "persons": [
        "athrva", "alice", "bob", "mary"  # List of trackable persons
    ],
    "objects": [
        "ball", "chair", "bottle", "cup"  # List of trackable objects
    ],
    "commands": ["track", "follow", "stop"]
}

def record_audio(duration=8, sample_rate=16000, output_file="/tmp/input_audio.wav"):
    print("Recording... Speak into the microphone.")
    print(f"You have {duration} seconds to speak...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    sf.write(output_file, audio, sample_rate)
    return output_file

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(file_path, audio_file.read()),
            model="whisper-large-v3-turbo",
            language="en",
        )
    return transcription.text

def parse_tracking_command(command):
    """Parse the command to extract tracking details."""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a command parser. Your job is to convert natural language commands into structured tracking commands.
                ANY command about tracking, following, or starting segmentation should be treated as a 'track' command.
                
                Examples:
                - "Start segmentation and recognition for atharva" → "track|person|atharva"
                - "Track person called atharva" → "track|person|atharva"
                - "Follow atharva" → "follow|person|atharva"
                - "Start tracking atharva" → "track|person|atharva"
                - "Stop" → "stop|none|none"
                
                Common variations to handle:
                - "atharva", "athrva", "a tharva" all mean "atharva"
                - "Start segmentation", "begin tracking", "start recognition" all mean "track"
                
                Always respond with one of these formats:
                1. action|target_type|target_name
                2. "INVALID" if you can't parse the command
                
                Be very lenient in interpretation - if it sounds like a tracking command, try to parse it."""
            },
            {
                "role": "user",
                "content": f"Parse this command: '{command}'"
            }
        ],
        model="llama3-8b-8192",
    )
    
    parsed = chat_completion.choices[0].message.content.strip()
    print(f"LLM parsed result: {parsed}")  # Debug print
    
    if parsed == "INVALID":
        return None
        
    # Validate parsed command
    try:
        action, target_type, target_name = parsed.split('|')
        
        # Handle stop command specially
        if action == "stop":
            return "stop|none|none"
            
        if action not in TRACKING_CONFIG['commands']:
            return None
        
        if target_type == "person" and target_name.lower() not in TRACKING_CONFIG['persons']:
            print(f"Unknown person: {target_name}")
            return None
        elif target_type == "object" and target_name.lower() not in TRACKING_CONFIG['objects']:
            print(f"Unknown object: {target_name}")
            return None
                
        return parsed
    except ValueError:
        print("Invalid command format")
        return None

def execute_tracking(parsed_command):
    """Execute the tracking command using ROS2."""
    try:
        action, target_type, target_name = parsed_command.split('|')
        
        # Start the segment_and_recognize node with parameters
        subprocess.Popen([
            'ros2', 'run', 'ergo', 'segment_and_recognize',
            '--ros-args',
            '-p', f'tracking_mode:={target_type}',
            '-p', f'target_object:={target_name if target_type == "object" else ""}',
            '-p', f'person_to_track:={target_name if target_type == "person" else ""}'
        ])

        # Start the tracking node
        subprocess.Popen([
            'ros2', 'run', 'ergo', 'track',
            '--ros-args',
            '-p', f'tracking_mode:={target_type}',
            '-p', f'target_object:={target_name if target_type == "object" else ""}',
            '-p', f'person_to_track:={target_name if target_type == "person" else ""}'
        ])

    except Exception as e:
        print(f"Error executing tracking command: {e}")

if __name__ == "__main__":
    while True:
        try:
            audio_file = record_audio()  # Now with 8 seconds duration
            transcript = transcribe_audio(audio_file)
            print(f"\nTranscription: {transcript}")

            if transcript.lower().strip() == "stop":
                print("Stopping all tracking...")
                subprocess.run(['ros2', 'topic', 'pub', '/cmd_vel', 'geometry_msgs/msg/Twist', 
                              '"{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"'])
                continue

            parsed_command = parse_tracking_command(transcript)
            print(f"Parsed Command: {parsed_command}")

            if parsed_command:
                print("Executing tracking command...")  # Debug print
                execute_tracking(parsed_command)
            else:
                print("Sorry, I didn't understand that command. Please try again.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
