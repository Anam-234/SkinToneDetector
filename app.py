import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import time
from utils.face_detection import detect_face
from utils.skin_extraction import extract_skin_region
from utils.color_analysis import get_dominant_color, color_to_hex, find_closest_skin_tone
from io import BytesIO
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Skin Tone Detector",
    page_icon="👤",
    layout="wide"
)

if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

# Load skin tone dataset
@st.cache_data
def load_skin_tone_data():
    return pd.read_csv("data/skin_tones.csv")

skin_tones_df = load_skin_tone_data()

# App title and description
st.title("Skin Tone Detector")
st.markdown("""
This application detects your skin tone and provides the closest matching 
skin tone name and hex color code. You can use your webcam in real-time or upload a photo.
""")

# Function to process an image and detect skin tone
def process_image(image):
    # Convert PIL Image to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create a copy of the image to draw on
    display_image = image.copy()
    
    # Detect face in the image
    face_landmarks = detect_face(image)
    
    # Initialize result variables
    current_hex = None
    current_tone_name = None
    result_image = None
    
    # Process the image if a face is detected
    if face_landmarks:
        # Extract skin region
        skin_mask, skin_region = extract_skin_region(image, face_landmarks)
        
        # Get dominant skin color if skin region is not empty
        if skin_region.size > 0:
            dominant_color = get_dominant_color(skin_region)
            hex_color = color_to_hex(dominant_color)
            
            # Find the closest skin tone name
            tone_name, distance = find_closest_skin_tone(hex_color, skin_tones_df)
            
            # Update results
            current_hex = hex_color
            current_tone_name = tone_name
            
            # Draw rectangle with detected color on the display image
            # Convert dominant_color to the correct format for OpenCV rectangle
            bgr_color = (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))
            cv2.rectangle(display_image, (10, 10), (50, 50), bgr_color, -1)
            
            # Write skin tone name on the display image
            cv2.putText(display_image, f"Tone: {tone_name}", (60, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw the face landmarks on the display image
            for landmark in face_landmarks:
                x, y = landmark
                cv2.circle(display_image, (int(x), int(y)), 1, (0, 255, 0), -1)
            
            # Convert display image to RGB for Streamlit
            result_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
    
    return current_hex, current_tone_name, result_image

# Mode selection tabs
tab1, tab2 = st.tabs(["Webcam Mode", "Photo Upload Mode"])

# Tab 1: Webcam Mode
with tab1:
    # Create columns for layout
    col1, col2 = st.columns([3, 2])
    
    # Placeholder for webcam feed
    with col1:
        webcam_placeholder = st.empty()
        
    # Placeholder for results
    with col2:
        st.subheader("Detected Skin Tone")
        skin_tone_name_webcam = st.empty()
        hex_code_webcam = st.empty()
        color_display_webcam = st.empty()
        stats_container = st.container()
        
        with stats_container:
            st.markdown("### Skin Tone Analysis")
            analysis_text = st.empty()
    
    # Button to start/stop webcam
    webcam_col1, webcam_col2 = st.columns(2)
    with webcam_col1:
        start_button = st.button("Start Webcam")
    
    if start_button:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            st.error("Could not open webcam. Please check your camera permissions or try the Photo Upload mode instead.")
        else:
            with webcam_col2:
                stop_button = st.button("Stop Webcam")
            
            # Create a placeholder for status message
            status_message = st.empty()
            
            # Initialize variables for skin tone tracking
            current_hex = None
            current_tone_name = None
            tone_count = {}
            start_time = time.time()
            frame_count = 0
            
            while cap.isOpened() and not stop_button:
                # Read frame from webcam
                ret, frame = cap.read()
                frame_count += 1
                
                if not ret:
                    st.error("Failed to capture image from webcam.")
                    break
                
                # Flip the frame horizontally for a more natural view
                frame = cv2.flip(frame, 1)
                
                # Detect face in the frame
                face_landmarks = detect_face(frame)
                
                # Process the frame if a face is detected
                if face_landmarks:
                    # Extract skin region
                    skin_mask, skin_region = extract_skin_region(frame, face_landmarks)
                    
                    # Get dominant skin color if skin region is not empty
                    if skin_region.size > 0:
                        dominant_color = get_dominant_color(skin_region)
                        hex_color = color_to_hex(dominant_color)
                        
                        # Find the closest skin tone name
                        tone_name, distance = find_closest_skin_tone(hex_color, skin_tones_df)
                        
                        # Update tracking
                        current_hex = hex_color
                        current_tone_name = tone_name
                        
                        # Count occurrences of skin tones
                        if tone_name in tone_count:
                            tone_count[tone_name] += 1
                        else:
                            tone_count[tone_name] = 1
                        
                        # Display results
                        skin_tone_name_webcam.markdown(f"**Name:** {tone_name}")
                        hex_code_webcam.markdown(f"**Hex Code:** {hex_color}")
                        color_display_webcam.markdown(
                            f'<div style="background-color: {hex_color}; width: 100%; height: 50px; border-radius: 5px;"></div>',
                            unsafe_allow_html=True
                        )
                        
                        # Draw rectangle with detected color on the frame
                        bgr_color = (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))
                        cv2.rectangle(frame, (10, 10), (50, 50), bgr_color, -1)
                        
                        # Write skin tone name on the frame
                        cv2.putText(frame, f"Tone: {tone_name}", (60, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                    # Draw the face landmarks on the frame
                    for landmark in face_landmarks:
                        x, y = landmark
                        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
                    
                    # Show the processed image with face landmarks
                    status_message.success("Face detected! Analyzing skin tone...")
                else:
                    status_message.warning("No face detected. Please position your face in the frame.")
                
                # Convert frame from BGR to RGB for displaying in Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                webcam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                
                # Update analysis statistics
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    fps = frame_count / elapsed_time
                    
                    # Find the most common skin tone if we have enough samples
                    most_common_tone = None
                    if tone_count:
                        most_common_tone = max(tone_count.items(), key=lambda x: x[1])[0]
                    
                    analysis_html = f"""
                    <div>
                        <p>Time elapsed: {elapsed_time:.1f} seconds</p>
                        <p>Frames processed: {frame_count}</p>
                        <p>FPS: {fps:.1f}</p>
                    </div>
                    """
                    
                    if most_common_tone:
                        analysis_html += f"<p>Most consistent skin tone: <strong>{most_common_tone}</strong></p>"
                    
                    analysis_text.markdown(analysis_html, unsafe_allow_html=True)
                
                # Sleep to reduce CPU usage
                time.sleep(0.01)
            
            # Release webcam when done
            cap.release()
            
            # Display final results
            if current_hex and current_tone_name and current_hex.startswith('#') and len(current_hex) == 7:
                st.success("Analysis complete!")
                st.markdown(f"### Final Results")
                st.markdown(f"**Determined Skin Tone:** {current_tone_name}")
                st.markdown(f"**Hex Color Code:** {current_hex}")
                
                # Safely extract BGR from hex
                try:
                    b = int(current_hex[5:7], 16)
                    g = int(current_hex[3:5], 16)
                    r = int(current_hex[1:3], 16)
                    color_bgr = (b, g, r)
                # Create downloadable image with results
                    result_img = np.zeros((200, 400, 3), dtype=np.uint8)
                    cv2.rectangle(result_img, (0, 0), (400, 200), color_bgr, -1)
                    cv2.putText(result_img, f"Skin Tone: {current_tone_name}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(result_img, f"Hex Code: {current_hex}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                          
                
                
                    # Convert to RGB for display
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                
                   # Display the result image
                    st.image(result_img_rgb, caption="Your Skin Tone Result", use_column_width=True)
                
                # Option to save as CSV
                    csv_data = pd.DataFrame({
                        'Skin Tone Name': [current_tone_name],
                        'Hex Code': [current_hex],
                        'RGB Value': [f"({r}, {g}, {b})"]         
                        })
                
                    st.download_button(
                       label="Download Results as CSV",
                       data=csv_data.to_csv(index=False),
                       file_name="skin_tone_results.csv",
                       mime="text/csv"   
                    )
                except ValueError as ve: 
                            st.error("Hex color code parsing failed. Please check the detected value.")
            else:
                st.warning("No skin tone was detected. Please try again with better lighting and face positioning or try the Photo Upload mode.")

# Tab 2: Photo Upload Mode
with tab2:
    st.markdown("""
    Upload a photo of yourself to detect your skin tone. For best results:
    - Choose a photo with good, even lighting
    - Make sure your face is clearly visible
    - Avoid photos with heavy filters or makeup
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([3, 2])
        
        # Read the uploaded image
        image = Image.open(uploaded_file)
        
        # Process the image
        hex_color, tone_name, result_image = process_image(image)
        
        # Display the processed image
        with col1:
            if result_image is not None:
                st.image(result_image, caption="Processed Image", use_column_width=True)
            else:
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.error("No face detected in the uploaded image. Please try another photo with a clearer view of your face.")
        
        # Display results
        with col2:
            if hex_color and tone_name:
                st.success("Analysis complete!")
                st.subheader("Detected Skin Tone")
                st.markdown(f"**Name:** {tone_name}")
                st.markdown(f"**Hex Code:** {hex_color}")
                st.markdown(
                    f'<div style="background-color: {hex_color}; width: 100%; height: 50px; border-radius: 5px;"></div>',
                    unsafe_allow_html=True
                )
                
                # Option to save as CSV
                csv_data = pd.DataFrame({
                    'Skin Tone Name': [tone_name],
                    'Hex Code': [hex_color],
                    'RGB Value': [f"({int(hex_color[1:3], 16)}, {int(hex_color[3:5], 16)}, {int(hex_color[5:7], 16)})"]
                })
                
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_data.to_csv(index=False),
                    file_name="skin_tone_results.csv",
                    mime="text/csv"
                )
            elif result_image is None:
                st.warning("No face detected in the image. Please upload a photo where your face is clearly visible.")
            else:
                st.warning("Could not determine skin tone. Please try another photo with better lighting.")

# Add information about the application
st.markdown("---")
st.markdown("""
### About This Application
This skin tone detector uses computer vision and machine learning techniques to identify your skin tone:

1. **Face Detection**: Using Mediapipe to accurately locate facial landmarks
2. **Skin Extraction**: Isolating skin pixels from the detected face region
3. **Color Analysis**: Using K-means clustering to find the dominant skin color
4. **Tone Matching**: Mapping the detected color to standardized skin tone names

For best results:
- Use natural, even lighting
- Face the camera directly or upload a clear front-facing photo
- Avoid wearing makeup if possible
- Position your face to fill a good portion of the frame
""")

# Add tips for accurate detection
with st.expander("Tips for Accurate Detection"):
    st.markdown("""
    - **Lighting**: Natural, diffused daylight provides the most accurate results
    - **Background**: Use a neutral background to avoid color reflections
    - **Distance**: Position yourself about 1-2 feet from the camera
    - **Angle**: Face the camera directly for the most accurate readings
    - **Multiple readings**: Take several readings and use the most consistent result
    """)
