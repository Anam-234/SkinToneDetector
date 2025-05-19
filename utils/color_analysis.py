import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import colorsys

def get_dominant_color(skin_region, n_clusters=5):
    """
    Extract the dominant color from the skin region using K-means clustering.
    
    Args:
        skin_region (numpy.ndarray): Masked skin region of the original image
        n_clusters (int): Number of clusters for K-means
        
    Returns:
        numpy.ndarray: RGB values of the dominant skin color
    """
    # Reshape the image to be a list of pixels
    pixels = skin_region.reshape(-1, 3)
    
    # Filter out black background pixels (from the mask)
    pixels = pixels[~np.all(pixels == 0, axis=1)]
    
    # If no valid pixels, return a default skin tone
    if len(pixels) == 0:
        return np.array([210, 170, 150])  # Default beige skin tone
    
    # Convert from BGR to RGB
    pixels = pixels[:, ::-1]
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(pixels)
    
    # Get the cluster centers and counts
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = np.bincount(labels)
    
    # Sort clusters by size
    sorted_indices = np.argsort(counts)[::-1]
    
    # Find the cluster with the most balanced RGB values (avoid too yellow or gray)
    for idx in sorted_indices:
        r, g, b = cluster_centers[idx]
        
        # Convert to HSV to check saturation and value
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        
        # Skip colors that are too saturated, too bright, too dark, or too yellow
        if 0.15 <= s <= 0.55 and 0.25 <= v <= 0.95:
            # Check if it's not too yellow (typically around 60 degrees in HSV)
            if not (0.11 <= h <= 0.2 and s > 0.4):
                return cluster_centers[idx].astype(np.uint8)
    
    # If no suitable color found, return the most common cluster
    return cluster_centers[sorted_indices[0]].astype(np.uint8)

def color_to_hex(rgb_color):
    """
    Convert RGB color to hex code.
    
    Args:
        rgb_color (numpy.ndarray): RGB color values
        
    Returns:
        str: Hex color code
    """
    r, g, b = rgb_color
    return f'#{r:02x}{g:02x}{b:02x}'

def rgb_to_lab(rgb_color):
    """
    Convert RGB color to LAB color space for better color comparison.
    
    Args:
        rgb_color (str or tuple): RGB color as hex string or tuple
        
    Returns:
        numpy.ndarray: LAB color values
    """
    # If input is hex string, convert to RGB tuple
    if isinstance(rgb_color, str) and rgb_color.startswith('#'):
        r = int(rgb_color[1:3], 16)
        g = int(rgb_color[3:5], 16)
        b = int(rgb_color[5:7], 16)
        rgb = np.array([[[r, g, b]]], dtype=np.uint8)
    else:
        rgb = np.array([[[rgb_color[0], rgb_color[1], rgb_color[2]]]], dtype=np.uint8)
    
    # Convert to LAB
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab[0, 0]

def color_distance(color1, color2):
    """
    Calculate the Delta E color distance between two colors in LAB space.
    
    Args:
        color1 (numpy.ndarray): First color in LAB space
        color2 (numpy.ndarray): Second color in LAB space
        
    Returns:
        float: Delta E distance value
    """
    # Calculate Euclidean distance in LAB space (Delta E)
    l1, a1, b1 = color1
    l2, a2, b2 = color2
    return np.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

def find_closest_skin_tone(detected_hex, skin_tones_df):
    """
    Find the closest matching skin tone name for a given hex color.
    
    Args:
        detected_hex (str): Hex color code of the detected skin tone
        skin_tones_df (pandas.DataFrame): DataFrame with skin tone data
        
    Returns:
        tuple: (skin_tone_name, distance) where:
               - skin_tone_name is the name of the closest matching skin tone
               - distance is the color distance between the detected tone and the matched tone
    """
    # Convert detected color to LAB
    detected_lab = rgb_to_lab(detected_hex)
    
    min_distance = float('inf')
    closest_tone = "Unknown"
    
    # Iterate through the skin tones dataset
    for _, row in skin_tones_df.iterrows():
        tone_name = row['name']
        tone_hex = row['hex_code']
        
        # Convert reference tone to LAB
        tone_lab = rgb_to_lab(tone_hex)
        
        # Calculate distance
        distance = color_distance(detected_lab, tone_lab)
        
        if distance < min_distance:
            min_distance = distance
            closest_tone = tone_name
    
    return closest_tone, min_distance
