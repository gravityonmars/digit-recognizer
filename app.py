import joblib
import numpy as np
from PIL import Image
import streamlit as st

st.title("Number Guessing ML App")

loaded = joblib.load("model.pkl")

if isinstance(loaded, tuple):
  model, scaler = loaded
else:
  model = loaded
  scaler = None

uploaded_file = st.file_uploader(
    "Upload a number image", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("L")
  st.image(image, caption="Uploaded Image", width=150)

  if st.button("Guess Number"):
    img_resized = image.resize((28, 28))
    img_array = np.array(img_resized)

    if np.mean(img_array) > 127:
      img_array = 255 - img_array

    img_array = img_array.reshape(1, -1)

    if scaler is not None:
      img_array = scaler.transform(img_array)

    prediction = model.predict(img_array)
    st.success(f"Predicted Number: {prediction[0]}")