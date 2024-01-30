import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
import pandas as pd

banner_image_path = "Schermafbeelding 2024-01-30 104931.png"
st.image(banner_image_path, use_column_width=True)

# Streamlit application
st.title("Discovery Week")
st.sidebar.header('Choose Input Method')

# Load the model and labels
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
recognized_animals = [
    "bever", "bonobo", "chimpansee", "dolfijn", "duif", "geit", "giraf",
    "goudvis", "hond", "kat", "leeuw", "luiaard", "mens", "nijlpaard",
    "orang-oetang", "oetang", "otter", "paard", "parkiet", "slang", "spin", "vos"
]

# Sidebar to choose input method
method = st.sidebar.radio('Select Input Method', ['Upload Image', 'Use Webcam'])

# Display recognized animals in the sidebar with each animal on a new row
st.sidebar.subheader('Recognized Animals')
for animal in recognized_animals:
    st.sidebar.write(f'- {animal}')

# Webcam feature
if method == 'Use Webcam':
    st.subheader("Webcam")

    # Create a VideoCapture object
    video_capture = cv2.VideoCapture(0)

    # Display webcam preview
    stframe = st.empty()
    screenshot_button = st.button("Take a screenshot")

    while True:
        _, frame = video_capture.read()

        # Check if the frame is empty
        if frame is None:
            st.warning("Unable to capture video.")
            break

        # Prepare the frame for the model
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Make prediction with the model when the "Take a screenshot" button is pressed
        if screenshot_button:
            st.subheader("Screenshot taken")
            st.image(frame, caption="Screenshot", channels="RGB", use_column_width=True)

            # Make prediction with the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_scores = prediction[0] * 100

            # Display results
            st.subheader("Result:")
            st.write(f"Prediction: {class_name[2:]}")
            st.write(f"Confidence Score: {confidence_scores[index]:.2f}%")

            # Additional feature: Display a bar chart of confidence scores
            df_confidence = pd.DataFrame({
                'Class': [class_names[i][2:] for i in range(len(class_names))],
                'Confidence Score': confidence_scores
            })
            st.subheader('Confidence Scores Bar Chart:')
            st.bar_chart(df_confidence.set_index('Class'))

            # Stop video capture when the user is done
            video_capture.release()
            break
        else:
            # Display live video feed
            stframe.image(frame, caption="Live video feed", channels="RGB", use_column_width=True)

# Upload Image feature
elif method == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        image_array = np.asarray(image.convert("RGB"))
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_scores = prediction[0] * 100

        # Additional feature: Display a bar chart of confidence scores
        st.subheader('Confidence Scores Bar Chart:')
        df_confidence = pd.DataFrame({
            'Class': [class_names[i][2:] for i in range(len(class_names))],
            'Confidence Score': confidence_scores
        })
        st.bar_chart(df_confidence.set_index('Class'))

        # Display results
        st.subheader("Result:")
        st.write(f"Prediction: {class_name[2:]}")
        st.write(f"Confidence Score: {confidence_scores[index]:.2f}%")

# Add a footer or any additional information as needed
st.sidebar.text("Made by Team 4")
