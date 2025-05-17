from facial_emotion_recognition import EmotionRecognition
import cv2

# Initialize the emotion recognition model to run on CPU
er = EmotionRecognition(device='cpu')

# Start video capture from the default webcam
cam = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    success, frame = cam.read()
    
    # Detect and annotate emotions on the frame
    frame = er.recognise_emotion(frame, return_type='BGR')
    
    # Display the frame with emotion annotations
    cv2.imshow("Frame", frame)
    
    # Wait for 1 ms and check if ESC key (27) is pressed to exit
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the webcam and close display window
cam.release()
cv2.destroyAllWindows()
