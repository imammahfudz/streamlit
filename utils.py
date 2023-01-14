from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Dropout, Dropout
from keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adam
from PIL import Image
from keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def gen_labels():
    labels = {0 : "Basalt", 1 : "Lapis Lazuli", 2 : "Obsidian", 3 : "Serpentine"}
    return labels

def preprocess(image):
    image = np.array(image.resize((300, 300), Image.ANTIALIAS))
    image = np.array(image, dtype='uint8')
    image = np.array(image)/255.0
    return image


def CNN_Model():
    input_shape = (300, 300, 3)
    num_classes=4
    model = Sequential()    
    model.add(Conv2D(32, kernel_size=(3,3), padding='Same', input_shape = input_shape, activation='relu'))
    model.add(MaxPool2D(pool_size=2)) 

    model.add(Conv2D(64, kernel_size = (3,3), padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=2)) 

    model.add(Conv2D(32, kernel_size = (3,3), padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=2)) 
    model.add(Flatten())

    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32,activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation = 'softmax'))
    return model  
