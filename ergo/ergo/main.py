import os
import sounddevice as sd
import soundfile as sf
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# INTENT_ACTIONS = {
#     "track": "track",
#     "exit_assistant": "exit"
# }

def record_audio(duration=5, sample_rate=16000, output_file="/tmp/input_audio.wav"):
    """Record audio for the specified duration."""
    print("Recording... Speak into the microphone.")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    sf.write(output_file, audio, sample_rate)
    return output_file

def transcribe_audio(file_path):
    """Transcribe audio using Groq's Whisper model."""
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(file_path, audio_file.read()),
            model="whisper-large-v3-turbo",
            language="en",
        )
    return transcription.text

def recognize_intent(command):
    """Recognize the intent of the command using Groq's LLM."""
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an assistant that maps user commands to intents."},
            {
                "role": "user",
                "content": f"Given this command: '{command}', what is the intent? Choose one from: track, exit_assistant."
            }
        ],
        model="llama3-8b-8192",
    )
    intent_response = chat_completion.choices[0].message.content.strip()
    
    if "track" in intent_response.lower():
        return "track"
    elif "exit_assistant" in intent_response.lower():
        return "exit_assistant"
    else:
        return "unknown"

def track():
    """Perform the tracking task."""
    print("Executing the 'track' function...")

if __name__ == "__main__":
    while True:
        try:
            audio_file = record_audio()

            transcript = transcribe_audio(audio_file)
            print(f"Transcription: {transcript}")

            intent = recognize_intent(transcript)
            print(f"Recognized Intent: {intent}")

            if intent == "track":
                track()
            elif intent == "exit_assistant":
                print("Exiting voice assistant...")
                break
            else:
                print("Sorry, I didn't understand that. Please try again.")

        except Exception as e:
            print(f"Error: {e}")
