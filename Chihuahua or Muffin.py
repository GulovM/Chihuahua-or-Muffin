import numpy as np
import pickle
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as tf_image

st.set_page_config(
    page_title="Chihuahua & Muffin clastering",
    layout="wide",
    initial_sidebar_state="expanded",
)

st_image = Image.open("Clastering.png")
st.image(st_image)

# Load kmeans and VGG16 model
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)
    
model = VGG16(weights='imagenet', include_top=False)

def predict_single_image(img_path):
    img = Image.open(img_path)
    img = img.convert("RGB").resize((224, 224))
    img_data = tf_image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    features = model.predict(img_data).flatten().astype(np.float64)
    prediction = kmeans.predict([features])
    
    if prediction[0] == 1:
        return "muffin"
    else:
        return "chihuahua"

def main():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        if st.button("Predict"):
            img_path = uploaded_file
            result = predict_single_image(img_path)
            st.success(f"Predicted label: {result}")

    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
