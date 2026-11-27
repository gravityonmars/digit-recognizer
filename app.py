import joblib
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from pathlib import Path

st.title("Number Guessing ML App")

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


@st.cache_resource
def load_model():
    loaded = joblib.load(MODEL_PATH)
    if isinstance(loaded, tuple):
        return loaded[0], loaded[1] if len(loaded) > 1 else None
    return loaded, None


try:
    model, scaler = load_model()
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a number image", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("L")
    except OSError:
        st.error("The uploaded file is not a valid image.")
        st.stop()

    st.image(image, caption="Uploaded Image", width=150)

    if st.button("Guess Number"):
        source_array = np.asarray(image, dtype=np.float32)
        border = np.concatenate(
            (
                source_array[0, :],
                source_array[-1, :],
                source_array[:, 0],
                source_array[:, -1],
            )
        )
        if np.mean(border) > 127:
            image = ImageOps.invert(image)

        image.thumbnail((20, 20), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), color=0)
        left = (28 - image.width) // 2
        top = (28 - image.height) // 2
        canvas.paste(image, (left, top))
        image = canvas
        img_array = np.asarray(image, dtype=np.float32)

        img_array = img_array.reshape(1, -1)
        if scaler is not None:
            img_array = scaler.transform(img_array)

        prediction = model.predict(img_array)
        st.success(f"Predicted Number: {prediction[0]}")