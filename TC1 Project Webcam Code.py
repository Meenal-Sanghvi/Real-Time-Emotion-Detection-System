import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    try:
        # Detect emotions using DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Get the emotion with the highest confidence
        emotion = result[0]['dominant_emotion']
        score = result[0]['emotion'][emotion]

        # Display the detected emotion and score
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Emotion: {emotion}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Score: {score:.2f}', (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        print(f"Analysis Error: {e}")
        cv2.putText(frame, "Error analyzing emotions", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
