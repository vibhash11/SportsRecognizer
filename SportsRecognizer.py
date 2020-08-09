import urllib.request
from PIL import Image
import Testing
import streamlit as st

preTrainedExists = True
model_path = './model/'
st.sidebar.title("About")

st.sidebar.info("Sports Recognizer.")
st.sidebar.info("Can only recognize between Baseball, Basketball, Cricket, Football, Hockey, Tennis.")

st.sidebar.title("Predict New Images")
user_input_url = st.sidebar.text_input("Enter URL to Image here!","")

st.write("Enter an image url from the left. You'll be able to view the image.")
st.write("When you're ready, submit a prediction on the left.") 

if user_input_url != "":
    path, _ = urllib.request.urlretrieve(user_input_url, "./tempFile")
    image = Image.open(path)
    st.image(image, caption = "Let's predict the sports!", width = 640)

if st.sidebar.button('Predict Sports'):
    prediction, confidence = Testing.predict(model_path, path) ### EDIT
    st.write("They are playing {}! Confidence = {:.2f}%.".format(prediction, confidence*100))