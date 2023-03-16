import numpy as np 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.keras.optimizers import Adam, RMSprop, SGD 
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.models import load_model

from PIL import Image, ImageOps
import glob

model=load_model('vgg_model.h5',compile=False)




def predict_image(imagepath, model):
    predict = load_img(imagepath, target_size = (224, 224))   
    predict_modified = img_to_array(predict)
    predict_modified = predict_modified / 255
    predict_modified = np.expand_dims(predict_modified, axis = 0)
    result = model.predict(predict_modified)
    image = Image.open(imagepath)
    plt.imshow(image)
    if result[0][0] >= 0.5:
        prediction = 'split'
        probability = result[0][0]
        print ("probability = " + str(probability))
        print("Prediction = " + prediction)
    else:
        prediction = 'no_split'
        probability = 1 - result[0][0]
        print ("probability = " + str(probability))
        print("Prediction = " + prediction)
    return prediction ,  round(probability, 2) 

