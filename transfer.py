import tensorflow as tf
import tensorflow_hub as hub

import tf_keras as keras
from tf_keras import Sequential 
from tf_keras import layers
from tf_keras import models
from tf_keras import losses
from tf_keras import utils

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import PIL
import PIL.Image
from sklearn.model_selection import train_test_split
img_height = 224
img_width = 224
num_classes = 4

# DATA PROCESSING
dir = "Data3/train/"
data = pd.read_csv("Data3/train/train.csv")

files = data['Image_Path']
labels = pd.get_dummies(data.astype(str), prefix="resistor_", columns=["Class"], dtype=int)
labels = labels.drop(["ID", "Image_Path"], axis=1)

X, y = [], []

print("begin data processing")
for i in range(10):
    print(i)
    img = PIL.Image.open(dir + files.iloc[i])
    img = img.resize((img_width,img_height))
    img = utils.img_to_array(img)
    X.append(img)
    y.append(labels.iloc[i])
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True)

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
pretrained_model_without_top_layer = hub.KerasLayer(
  feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

# hub_layer_wrapper = layers.Lambda(lambda x: pretrained_model_without_top_layer(x))
model = Sequential([
  # hub_layer_wrapper,
  # pretrained_model_without_top_layer,
  layers.Dense(num_classes)
])

model.compile(
  optimizer="adam",
  loss=losses.CategoricalCrossentropy(from_logits=True),
  metrics=['acc']
)
print(X_train.shape)
print(y_train.shape)
model.fit(X_train, y_train, epochs=1)
print("success")
# model.evaluate(X_test, y_test)
