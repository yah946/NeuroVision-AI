
import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf


# =====================================
# 1. Configuration
# =====================================
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🧠",
    layout="centered"
)

MODEL_PATH = "model/best_model.keras"
DATA_DIR = "data/raw"
IMG_SIZE = (224, 224)


# =====================================
# 2. Load model and class names
# =====================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    class_names = sorted(
        folder
        for folder in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, folder))
    )

    if not class_names:
        raise ValueError("No class folders found in the dataset.")

    return class_names


# =====================================
# 3. Preprocess image using OpenCV
# =====================================
def preprocess_image(image_bytes):
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Invalid or corrupted image.")

    # Convert BGR to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize to CNN input dimensions
    image = cv2.resize(
        image,
        IMG_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


# =====================================
# 4. Prediction function
# =====================================
def predict_image(image_bytes, model, class_names):
    image = preprocess_image(image_bytes)

    predictions = model.predict(image, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))

    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    return predicted_class, confidence, predictions


# =====================================
# 5. Check model file
# =====================================
if not os.path.isfile(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.write("Current working directory:", os.getcwd())
    st.stop()


# =====================================
# 6. Load resources
# =====================================
try:
    model = load_model()
    class_names = load_class_names()

    if len(class_names) != model.output_shape[-1]:
        st.error(
            "The number of classes in data/raw does not match "
            "the number of model output neurons."
        )
        st.stop()

except Exception as error:
    st.error(f"Unable to load resources: {error}")
    st.stop()


# =====================================
# 7. Streamlit interface
# =====================================
st.title("🧠 CNN Image Classifier")

st.write(
    "Upload an image to test your trained "
    "Convolutional Neural Network."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp"]
)


if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.getvalue()

        # Decode image for display
        display_image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if display_image is None:
            raise ValueError("Invalid or corrupted image.")

        display_image = cv2.cvtColor(
            display_image,
            cv2.COLOR_BGR2RGB
        )

        st.subheader("Uploaded image")
        st.image(
            display_image,
            caption="Input image",
            width="stretch"
        )

        if st.button("Predict image", type="primary"):
            with st.spinner("Predicting..."):
                predicted_class, confidence, predictions = (
                    predict_image(
                        image_bytes,
                        model,
                        class_names
                    )
                )

            st.success("Prediction completed!")

            st.subheader("Prediction result")
            st.write(f"**Predicted class:** {predicted_class}")

            st.metric(
                label="Confidence",
                value=f"{confidence:.2%}"
            )

            st.subheader("Class probabilities")

            for index, class_name in enumerate(class_names):
                probability = float(predictions[index])

                st.write(
                    f"**{class_name}** — {probability:.2%}"
                )
                st.progress(probability)

    except (ValueError, cv2.error) as error:
        st.error(f"Image processing error: {error}")

    except Exception as error:
        st.error(f"Prediction failed: {error}")
