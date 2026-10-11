import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("Edge Detection Using Basic Convolution")
st.write("Upload an image to apply horizontal, vertical, and combined convolution filters for edge detection.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    st.subheader("Original & Grayscale Image")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(gray, caption="Grayscale Image", use_container_width=True, channels="GRAY")

    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    kernel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    edges_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    edges_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
    
    edges_x_abs = cv2.convertScaleAbs(edges_x)
    edges_y_abs = cv2.convertScaleAbs(edges_y)

    combined_edges = cv2.magnitude(edges_x, edges_y)
    combined_edges_abs = cv2.convertScaleAbs(combined_edges)

    st.subheader("Convolution Filter Results")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.image(edges_x_abs, caption="Horizontal Edges (Kernel X)", use_container_width=True, channels="GRAY")
    with col4:
        st.image(edges_y_abs, caption="Vertical Edges (Kernel Y)", use_container_width=True, channels="GRAY")
    with col5:
        st.image(combined_edges_abs, caption="Combined Edges (Magnitude)", use_container_width=True, channels="GRAY")
        
else:
    st.info("Please upload an image file (JPG, PNG) to start edge detection.")