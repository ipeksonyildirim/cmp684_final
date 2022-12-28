# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 07:22:19 2022

@author: isony
"""
# load_model_sample.py
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(64, 64))
    img_tensor = image.img_to_array(img)                   
    img_tensor = np.expand_dims(img_tensor, axis=0)        
    img_tensor /= 255.                                      

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor
def tf_load_process_image(filename):
  img_size = 224
  # Load image (in PIL image format by default)
  image_original = load_img(filename, target_size=(img_size, img_size))
  # Convert from numpy array
  image_array = img_to_array(image_original)
  # Expand dims to add batch size as 1
  image_batch = np.expand_dims(image_array, axis=0)
  # Preprocess image
  image_preprocessed = tf.keras.applications.vgg16.preprocess_input(image_batch)
  return  image_preprocessed

# load model
model = load_model("model.h5")

def predict_single_img(img_path):
    new_image =  tf_load_process_image(img_path) # load a single image
    pred_result = np.argmax(model.predict(new_image)) # check prediction
    print(pred_result)
    return pred_result
    

def send_prediction_result(pred_res):
    if pred_res == 0:
        prediction = "NO DIABETIC RETINOPATHY"
    elif pred_res == 1:
        prediction = "MILD DIABETIC RETINOPATHY"
    elif pred_res == 2:
        prediction = "MODERATE DIABETIC RETINOPATHY"   
    elif pred_res == 3:
        prediction = "SEVERE DIABETIC RETINOPATHY"
    elif pred_res == 4:
        prediction = "PROLIFERATIVE DIABETIC RETINOPATHY"
            
    return prediction;

