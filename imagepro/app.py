import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Image Processing Functions
def apply_smoothing(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_contrast_stretching(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    min_val = np.min(image)
    max_val = np.max(image)
    stretched = (image - min_val) * (255 / (max_val - min_val))
    return np.array(stretched, dtype=np.uint8)

def apply_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(image, 100, 200)

# Streamlit UI
st.title("🖼 Image Enhancement & Processing")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption="Original Image", use_column_width=True)

    option = st.selectbox("Choose an enhancement technique", [
        "Smoothing",
        "Sharpening",
        "Contrast Stretching",
        "Edge Detection"
    ])

    if st.button("Apply"):
        if option == "Smoothing":
            processed_image = apply_smoothing(image_np)
            display_mode = "RGB"
        elif option == "Sharpening":
            processed_image = apply_sharpening(image_np)
            display_mode = "RGB"
        elif option == "Contrast Stretching":
            processed_image = apply_contrast_stretching(image_np)
            display_mode = "GRAY"
        elif option == "Edge Detection":
            processed_image = apply_edge_detection(image_np)
            display_mode = "GRAY"

        st.image(processed_image, caption="Processed Image", use_column_width=True, channels=display_mode)
