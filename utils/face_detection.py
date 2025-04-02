import cv2
import mediapipe as mp
import numpy as np

# Initialize the MediaPipe face mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define indices for face landmarks that correspond to cheek and forehead regions
# These are areas that typically represent natural skin tone
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# Cheeks and forehead indices
CHEEKS_FOREHEAD = [
    # Right cheek
    50, 101, 36, 206, 207, 187, 123, 116, 143, 156, 70, 63, 117, 118, 119, 47, 114,
    # Left cheek
    425, 427, 411, 280, 346, 340, 349, 345, 352, 376, 433, 434, 430, 431, 262, 428,
    # Forehead
    251, 284, 298, 333, 299, 296, 336, 285, 417, 168, 197, 5, 4, 75, 97, 2, 326, 305
]

def detect_face(frame):
    """
    Detect face in the input frame and return the facial landmarks.
    
    Args:
        frame (numpy.ndarray): Input image frame from webcam
        
    Returns:
        list: List of facial landmarks (x, y coordinates) if face is detected,
              None otherwise
    """
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and get face landmarks
    results = face_mesh.process(rgb_frame)
    
    # Initialize an empty list to store landmarks
    landmarks = []
    
    # Check if face landmarks are detected
    if results.multi_face_landmarks:
        # Get the first detected face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get image dimensions
        h, w, _ = frame.shape
        
        # Extract cheeks and forehead landmarks
        for idx in CHEEKS_FOREHEAD:
            # Convert normalized coordinates to pixel coordinates
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            
            # Add coordinates to landmarks list
            landmarks.append((x, y))
    
    return landmarks if landmarks else None
