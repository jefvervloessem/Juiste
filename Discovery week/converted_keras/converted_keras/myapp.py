import streamlit as st
import cv2
from PIL import Image, ImageOps
import requests
import numpy as np
from io import BytesIO
from keras.models import load_model

# Streamlit-applicatie
st.title("Discovery Week")

# Dierenherkenning met Teachable Machine
st.header("Dierenherkenning")

# Laden van het model en labels
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Uploaden van de afbeelding of gebruik de webcam
option = st.radio("Kies een optie:", ("Upload een afbeelding", "Gebruik de webcam"))

if option == "Upload een afbeelding":
    # Uploaden van de afbeelding
    uploaded_file = st.file_uploader("Upload een afbeelding", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Weergave van de geüploade afbeelding
        image = Image.open(uploaded_file)
        st.image(image, caption="Geüploade afbeelding", use_column_width=True)

        # Voorbereiden van de afbeelding voor het model
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        image_array = np.asarray(image.convert("RGB"))  # Convert image to RGB explicitly
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Voorspelling maken met het model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Resultaten weergeven
        st.subheader("Resultaat:")
        st.write(f"Voorspelling: {class_name[2:]}")
        st.write(f"Zekerheid: {confidence_score:.2f}")

elif option == "Gebruik de webcam":
    # Webcamfunctie
    st.subheader("Webcam")

    # Create a VideoCapture object
    video_capture = cv2.VideoCapture(0)

    # Display webcam preview
    stframe = st.empty()
    screenshot_button = st.button("Neem screenshot")

    while True:
        _, frame = video_capture.read()

        # Voorbereiden van het frame voor het model
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Voorspelling maken met het model als de knop "Neem screenshot" wordt ingedrukt
        if screenshot_button:
            st.subheader("Screenshot genomen")
            st.image(frame, caption="Screenshot", channels="RGB", use_column_width=True)

            # Voorspelling maken met het model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Resultaten weergeven
            st.subheader("Resultaat:")
            st.write(f"Voorspelling: {class_name[2:]}")
            st.write(f"Zekerheid: {confidence_score:.2f}")

            # Stop de videocapture wanneer de gebruiker klaar is
            video_capture.release()
            break
        else:
            # Display live video feed
            stframe.image(frame, caption="Live video feed", channels="RGB", use_column_width=True)