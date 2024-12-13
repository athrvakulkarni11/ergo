import pickle
import cv2
import os
import numpy as np

# Create a directory to store images
base_dir = 'person_images'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Save face recognizer state to a file
def save_embeddings(recognizer):
    recognizer.write('face_recognizer_model.yml')
    print("Face recognizer model has been saved.")

# Load face recognizer state from a file or initialize a new recognizer
def load_or_initialize_embeddings():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists('face_recognizer_model.yml'):
        recognizer.read('face_recognizer_model.yml')
        print("Loaded existing face recognizer model.")
    else:
        print("No existing model found. Starting fresh.")
    return recognizer

# Capture images of persons and save them
def capture_images():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Enter the person's name (or 'q' to quit): ")

    while True:
        person_name = input("Name: ").strip()
        if person_name.lower() == 'q':
            break

        person_dir = os.path.join(base_dir, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        print(f"Capturing images for {person_name}. Press 'c' to capture, 'q' to stop.")
        image_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow(f"Capturing - {person_name}", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and len(faces) > 0:
                x, y, w, h = faces[0]
                face_image = frame[y:y + h, x:x + w]
                image_path = os.path.join(person_dir, f"{person_name}_{image_counter}.jpg")
                cv2.imwrite(image_path, face_image)
                print(f"Image saved: {image_path}")
                image_counter += 1
            elif key == ord('q'):
                break

    cv2.destroyAllWindows()
    cap.release()
def generate_face_embeddings():
    recognizer = load_or_initialize_embeddings()
    labels = []
    images = []
    label_map = {}
    current_label = 0

    print("Generating face embeddings...")
    for person_name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            labels.append(current_label)
            images.append(image)

        label_map[current_label] = person_name
        current_label += 1

    if images:
        recognizer.train(images, np.array(labels))
        recognizer.write('face_recognizer_model.yml')  # Save the trained recognizer
        print("Face recognizer model has been saved.")

    # Save the label map for mapping IDs to names
    with open('label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)

    print("Face embeddings and label map have been saved.")
def recognize_faces():
    recognizer = load_or_initialize_embeddings()

    # Load the label map
    with open('label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_image = gray[y:y + h, x:x + w]
            face_image = cv2.resize(face_image, (200, 200))  # Resize for consistency

            label, confidence = recognizer.predict(face_image)
            name = label_map.get(label, "Unknown")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main function
def main():
    print("Choose an option:")
    print("1. Capture images of a person")
    print("2. Generate face embeddings")
    print("3. Recognize faces in real-time")
    print("q. Quit")

    while True:
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            capture_images()
        elif choice == '2':
            generate_face_embeddings()
        elif choice == '3':
            recognize_faces()
        elif choice.lower() == 'q':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
