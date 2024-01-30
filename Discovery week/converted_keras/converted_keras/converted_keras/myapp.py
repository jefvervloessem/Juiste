import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model

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

if method == 'Upload Image':
    # Upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for the model
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        image_array = np.asarray(image.convert("RGB"))
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Make prediction with the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Display results
        st.subheader("Result:")
        st.write(f"Prediction: {class_name[2:]}")
        st.write(f"Confidence Score: {confidence_score:.2f}")

elif method == 'Use Webcam':
    # Webcam feature
    st.subheader("Webcam")

    # Create a VideoCapture object
    video_capture = cv2.VideoCapture(0)

    # Display webcam preview
    stframe = st.empty()
    screenshot_button = st.button("Take a screenshot")

    while True:
        _, frame = video_capture.read()

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
            confidence_score = prediction[0][index]

            # Display results
            st.subheader("Result:")
            st.write(f"Prediction: {class_name[2:]}")
            st.write(f"Confidence Score: {confidence_score:.2f}")

            # Stop video capture when the user is done
            video_capture.release()
            break
        else:
            # Display live video feed
            stframe.image(frame, caption="Live video feed", channels="RGB", use_column_width=True)

# Add a footer or any additional information as needed
st.sidebar.text("Made by Team 4")
