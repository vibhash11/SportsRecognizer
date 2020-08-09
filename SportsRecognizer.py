import urllib.request
from PIL import Image as PImage
import Testing
import streamlit as st
from torchvision import transforms

def main():
    preTrainedExists = True
    model_path = './model/'
    st.sidebar.title("About")

    st.sidebar.info("Sports Recognizer.")
    st.sidebar.info("Can only recognize between Baseball, Basketball, Cricket, Football, Hockey, Tennis.")

    st.sidebar.title("Predict New Images")
    user_input_url = st.sidebar.text_input("Enter URL to Image here!","")

    st.write("Enter an image url from the left. You'll be able to view the image.")
    st.write("When you're ready, submit a prediction on the left.") 

    image_tensor = ""
    if user_input_url != "":
        try:
            image = PImage.open(urllib.request.urlopen(user_input_url))
            image_tensor = transforms.ToTensor()(image)
            st.image(image, caption = "Let's predict the sports!", width = 640)
            if st.sidebar.button('Predict Sports'):
                prediction, confidence = Testing.predict(model_path, image_tensor) ### EDIT
                st.write("They are playing {}! Confidence = {:.2f}%.".format(prediction, confidence*100))
        except Exception as e:
            st.write("Invalid Input: " + str(e))
            

if __name__ == '__main__':
    main()
