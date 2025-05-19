import cv2
import numpy as np

def extract_skin_region(frame, face_landmarks):
    """
    Extract skin regions from the detected face using color thresholding.
    
    Args:
        frame (numpy.ndarray): Input image frame from webcam
        face_landmarks (list): List of facial landmarks (x, y coordinates)
        
    Returns:
        tuple: (skin_mask, skin_region) where:
               - skin_mask is a binary mask of skin pixels
               - skin_region is the masked skin area of the original image
    """
    # Convert the frame to HSV color space for better skin detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a face region mask using the landmarks
    h, w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create points array for cv2.fillConvexPoly
    points = np.array(face_landmarks, dtype=np.int32)
    
    # Fill the region defined by the landmarks
    cv2.fillConvexPoly(mask, points, 255)
    
    # Define skin color range in HSV space
    # These thresholds work for a variety of skin tones
    # Lower HSV threshold for skin detection
    lower_threshold = np.array([0, 20, 70], dtype=np.uint8)
    upper_threshold = np.array([20, 180, 255], dtype=np.uint8)
    
    # Create a second range for skin detection (for darker skin tones)
    lower_threshold2 = np.array([170, 20, 70], dtype=np.uint8)
    upper_threshold2 = np.array([180, 180, 255], dtype=np.uint8)
    
    # Create skin masks using the thresholds
    skin_mask1 = cv2.inRange(hsv_frame, lower_threshold, upper_threshold)
    skin_mask2 = cv2.inRange(hsv_frame, lower_threshold2, upper_threshold2)
    
    # Combine the two skin masks
    skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
    
    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    
    # Combine with the face region mask to only consider skin within the face region
    skin_mask = cv2.bitwise_and(skin_mask, mask)
    
    # Apply the mask to the original frame to extract skin regions
    skin_region = cv2.bitwise_and(frame, frame, mask=skin_mask)
    
    return skin_mask, skin_region
