import tensorflow as tf
import tensorflow_hub as hub

import keras
from keras import Sequential 
from keras import layers
from keras import models

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import PIL
import PIL.Image
import pathlib

from sklearn.model_selection import train_test_split

img_height = int(3024/32) # 189
img_width = int(4032/32) # 252
num_classes = 4

# DATA PROCESSING
dir = "Data3/train/"
data = pd.read_csv("Data3/train/train.csv")

files = data['Image_Path']
labels = pd.get_dummies(data.astype(str), prefix="resistor_", columns=["Class"], dtype=int)
labels = labels.drop(["ID", "Image_Path"], axis=1)

X, y = [], []

print("begin data processing")
for i in range(len(files)):
    print(i)
    img = PIL.Image.open(dir + files.iloc[i])
    img = img.resize((img_width,img_height))
    img = keras.utils.img_to_array(img)
    X.append(img)
    y.append(labels.iloc[i])
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(files, labels, shuffle = True)

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

model = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

model.fit(X_train, y_train, epochs=5)

model.predict(X_test,y_test)
