# -*- coding: utf-8 -*-
"""pruebaStreamlit.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/125B9aHtHwYj0Zqr1KGHZ8zGDhJVcaI_2
"""


from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
import streamlit as st
from PIL import Image
from google.colab import drive
from skimage.transform import resize

names=["Elefante", "Mariposa","Vaca", "Oveja","Ardilla"]

def model_prediccion(img, model):
  img_resize=resize(img, (224,224))
  x= preprocess_input(img_resize*225)
  x=np.expand_dims(x, axis=0)

  preds=model.predict(x)
  return preds

def main():
  modelo=load_model("/content/drive/MyDrive/DATASET/modelo_animales1.h5/")
  st.title("Clasificador de Animales")
  img_file_buffer=st.file_uploader("Cargar una imagen", type=["png", "jpg", "jpeg"])
  if img_file_buffer is not None:
    image=np.array(Image.open(img_file_buffer))
    st.image(image, caption="Imagen", use_column_width=False)
  if st.button("Predicción"):
    predict=model_prediccion(image, modelo)
    st.success("La clase es:".format(names[np.argmax(predict)]))
