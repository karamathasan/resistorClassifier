import tensorflow as tf
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
# train.csv and test.csv had to be altered. \ was converted to /
print("\n\n\n\n")

# img_height = int(3024/32) # 189
# img_width = int(4032/32) # 252
img_height = 224
img_width = 224 
num_classes = 4

# model = Sequential([
#   layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Dropout(0.2),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Dropout(0.4),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
model.compile(\
    optimizer='adam',
    loss = keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy()]
    )

dir = "Data3/train/"
data = pd.read_csv("Data3/train/train.csv")

files = data['Image_Path']
labels = pd.get_dummies(data.astype(str), prefix="resistor_", columns=["Class"], dtype=int)
labels = labels.drop(["ID", "Image_Path"], axis=1)

# load all images 
print("loading images")
imgs = []
for i in range(len(files)):
# for i in range(len(files)//4):
    print(i)
    img = PIL.Image.open(dir + files.iloc[i])
    img = img.resize((img_width,img_height))
    img = keras.utils.img_to_array(img)
    imgs.append(img)
imgs = np.array(imgs)
epochs = 20 # usually 10
batch_size = 16 # usually 64

# labels = labels.iloc[0:len(labels)//4]
X_train, X_test, y_train, y_test = train_test_split(imgs, labels, shuffle = True)

# /////// ITERATIVE FITTING ///////
for j in range(epochs):
    print(f"epoch {j+1}/{epochs}")
    X_train, X_test, y_train, y_test = train_test_split(imgs, labels, shuffle = True)
    X = []
    y = []
    for i in range(len(X_train)):
        if (i % batch_size == 0 or i == len(X_train)-1) and i != 0: 
            X = np.array(X)
            y = np.array(y)
            print(f"        {X.shape}")
            model.fit(X,y, batch_size=batch_size)
            X = []
            y = []
        X.append(imgs[i])
        y.append(y_train.iloc[i])

# //////// TESTING /////////
# test_dir = "Data3/test/"

# test performed on split
score = 0
for i in range(len(X_test)):
    img = X_test[i]
    prediction = model.predict(np.expand_dims(keras.utils.img_to_array(img,dtype=float),axis = 0))
    true_class = labels.columns[np.argmax(y_test.iloc[i])]
    predicted_class = labels.columns[np.argmax(prediction)]
    # print(f"    location: {X_test[i]}")
    print(f'    true class: {true_class}, predicted class: {predicted_class}')
    # if (y_test.iloc[i].to_numpy().all() == prediction.all()):
    if (true_class == predicted_class):
        score+=1
print(f"accuracy: {100 * (score / len(X_test))}%")


