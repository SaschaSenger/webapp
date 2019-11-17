from app import app
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras import backend as K
import keras
import os
import tensorflow as tf
from keras.models import model_from_json

import numpy as np

def loadCategoriser():
    # load json and create model
    json_file = open('./model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    # load weights into new model
    classifier.load_weights("model.h5")
    print("Loaded model from disk")

    class_indices = ["10_Choco_Haps", "11_K_Classic_Mehl", "12_K_Classic_Zucker",
                 "1_Haribo_Goldbaer", "2_Chock_IT", "4_Kornflakes",
                 "5_K_Classic_Paprika_Chips", "6_Birnen_Dose", "7_Pfirsiche_Dose",
                 "9_Aprikosen_Dose"]

    test_image = image.load_img('static/img/Compare.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    print(result.round(5))
    return class_indices[np.argmax(result)]
