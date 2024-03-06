# Brain Tumor Segmentation App

![Brain Tumor Segmentation](https://via.placeholder.com/800x300)

## Overview
This Streamlit application segments brain tumors from MRI images using a trained U-Net model. Users can upload an MRI image, and the app will generate a predicted mask highlighting the tumor region.

## Installation
To run the app, follow these steps:

1. Clone this repository.
2. Install the required Python packages:
    ```bash
    pip install streamlit numpy scikit-image opencv-python tensorflow matplotlib
    ```

## Usage
1. Run the Streamlit app using the following command:
    ```bash
    streamlit run app.py
    ```
2. Once the app is running, upload an MRI image (in TIFF, JPEG, or PNG format).
3. The app will display the original image, the predicted mask, and an overlay of the image with the predicted mask.
4. Use the slider to adjust the transparency of the mask overlay.

## About the Author
This app was developed by Yash Patil.

## About
This application segments brain tumors from MRI images using a trained U-Net model. Upload an MRI image to get started!
