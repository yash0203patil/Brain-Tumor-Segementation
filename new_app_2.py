import streamlit as st
import numpy as np
from skimage import io
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import load_model
# model = load_model('model.h5')

# Load the model
model = load_model("unet_brain_tumor_segmentation.h5" , compile = False)

def segment_image(image_path):
    # Creating a empty array of shape 1,256,256,3
    X = np.empty((1, 256, 256, 3))
    
    # Read the image
    img = io.imread(image_path)
    
    # Resizing the image and converting them to array of type float64
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype=np.float64)
    
    # Standardising the image
    img -= img.mean()
    img /= img.std()
    
    # Converting the shape of image from 256,256,3 to 1,256,256,3
    X[0,] = img

    # Make prediction of mask
    predict = model.predict(X)
    pred = np.array(predict).squeeze().round()

    return img, pred

# Streamlit UI
st.title('Brain Tumor Segmentation')

uploaded_file = st.file_uploader("Choose an image...", type=["tif", "tiff", "jpg", "jpeg", "png"])

if uploaded_file:
    img, pred = segment_image(uploaded_file)

    # Plotting
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    
    
    ax[0].imshow(img)
    ax[0].set_title("Input Image")
    ax[0].axis('off')

    ax[1].set_title("Predicted Mask")
    ax[1].imshow(pred)
    ax[1].axis('off')

    ax[2].imshow(img)
    ax[2].imshow(pred, alpha=0.5, cmap='jet')
    ax[2].set_title('Image with Predicted Mask')
    ax[2].axis('off')

    st.pyplot(fig)

    # Slider to control the alpha (transparency) of the mask
    alpha = st.slider("Adjust mask transparency:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.imshow(pred, alpha=alpha, cmap='jet')
    ax.set_title('Adjusted Image with Predicted Mask')
    ax.axis('off')
    
    st.pyplot(fig)

st.write("## About")
st.write("This application segments brain tumors from MRI images using a trained U-Net model. "
         "Upload an MRI image to get started!")
