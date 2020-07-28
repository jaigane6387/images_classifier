# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 08:04:26 2020

@author: jai ganesh
"""


#importing required libraries
import tensorflow
import io
import os
import numpy as np
import keras
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.xception import preprocess_input
from keras.applications.xception import decode_predictions
from keras.models import load_model
import streamlit as st
import h5py

#loading model

model = load_model('xception_model')
model._make_predict_function() 

def predict(image1): 
    image = load_img(image1, target_size=(299, 299)) #for xception it expects 299x299 size
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    return label 

 #main page
def main():
    st.title("Image Guessing Application")

    html_temp2 ="""
    <div style="padding:5px">
    <h6 style="color:green;text-align:right;font-weight:bold;font-style:verdana;">created by &copy;jai ganesh Nagidi</h6>
    </div>
    """
    st.markdown(html_temp2,unsafe_allow_html=True)
    html_temp="""
    <div style="background-color:purple;padding:10px">
    <h2 style="color:white;text-align:center;">Let's Classifiy &#128522;&#128526;</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding',False)
    uploaded_file = st.file_uploader("Choose an image...")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file,mode='r')
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    if st.button("Predict"):
        if uploaded_file is None:
            raise Exception("image not uploaded, please refresh page and upload the image")
        st.write("")
        st.write("Classifying...")
        with open(uploaded_file, 'rb') as f:
            label = predict(f)
        st.write('%s (%.2f%%)' % (label[1], label[2]*100))
    hide_streamlit_style ="""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    html_temp3="""
    <p>This application can able to detect 1000+ different objects.This application was developed by &copy; Jai Ganesh Naigidi<br>
        you can connect with him :<a href="https://www.linkedin.com/in/jaiganesh-nagidi-4205a4181/">Let's connect</a>
    </p>
    """
    if st.button("About"):
        st.markdown(html_temp3,unsafe_allow_html=True)

if __name__=='__main__':
    main()
    
