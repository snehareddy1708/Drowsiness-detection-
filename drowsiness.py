import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import pygame

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Load VideoCapture
cap = cv2.VideoCapture(0)
# Constants
EAR_THRESHOLD = 2
EAR_CONSEC_FRAMES = 48
COUNTER = 0

# Initialize Pygame Mixer
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.wav')

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks on face
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1),
                )

                # Extract eye landmarks
                left_eye = [tuple([landmark.x, landmark.y]) for landmark in face_landmarks.landmark[263:276]]
                right_eye = [tuple([landmark.x, landmark.y]) for landmark in face_landmarks.landmark[23:36]]
                # Calculate EAR for both eyes
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                print(f"EAR Value : {ear}")  # Print the calculated EAR value
                # Check if EAR is below threshold
                if ear < EAR_THRESHOLD:
                    COUNTER += 1
                    if COUNTER >= EAR_CONSEC_FRAMES:
                        # Drowsiness detected, trigger an alert
                        alarm_sound.play()
                        text='DRIVER IS DROWSY'
                        font=cv2.FONT_HERSHEY_COMPLEX
                        org=(50,50)
                        font_scale=1
                        color=(0,0,255)
                        thickness=2
                        cv2.putText(frame,text,org,font,font_scale,color, thickness,cv2.LINE_AA)
                        # Add your alert action here (e.g., sound alarm, send notification)

                else:
                    COUNTER = 0
                    alarm_sound.stop()
                    text='DRIVER IS NOT DROWSY'
                    font=cv2.FONT_HERSHEY_COMPLEX
                    org=(50,50)
                    font_scale=1
                    color=(0,0,255)
                    thickness=2
                    cv2.putText(frame,text,org,font,font_scale,color, thickness,cv2.LINE_AA)

        cv2.imshow('Drowsiness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
