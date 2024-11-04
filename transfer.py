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
from env import SECRET
from matplotlib import pyplot as plt

img_height = 224
img_width = 224
num_classes = 4

# DATA PROCESSING
dir = "Data3/train/"
data = pd.read_csv("Data3/train/train.csv")

files = data['Image_Path']
labels = pd.get_dummies(data.astype(str), prefix="resistor_", columns=["Class"], dtype=int)
labels = labels.drop(["ID", "Image_Path"], axis=1)

# batch_size = 32
# validation_split = 0.2
# train_ds = keras.preprocessing.image_dataset_from_directory(
#     dir,
#     validation_split=validation_split,
#     subset="training",
#     seed=42,
#     image_size=(img_height, img_width),
#     batch_size=batch_size
# )

# # Load validation/testing data with 20% split
# val_ds = keras.preprocessing.image_dataset_from_directory(
#     dir,
#     validation_split=validation_split,
#     subset="validation",
#     seed=42,
#     image_size=(img_height, img_width),
#     batch_size=batch_size
# )

X, y = [], []
print("begin data processing")
for i in range(len(files)):
    print(i)
    img = PIL.Image.open(dir + files.iloc[i])
    img = img.resize((img_width,img_height))
    img = utils.img_to_array(img)
    X.append(img)
    y.append(labels.iloc[i])
X = np.array(X, dtype=np.float32)
X /= 255
y = np.array(y, dtype=np.float32)

# X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, stratify=labels.iloc[0:len(y)])
train_ds = tf.data.Dataset.from_tensor_slices((X_train,y_train)).batch(32)
val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
pretrained_model_without_top_layer = hub.KerasLayer(
  feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

augment = Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
])

model = Sequential([
  augment,
  pretrained_model_without_top_layer,
  layers.Dense(num_classes)
])

model.compile(
  optimizer="adam",
  loss=losses.CategoricalCrossentropy(from_logits=True),
  metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")]
)

epochs = 40
# history = model.fit(X_train, y_train, validation_data=val_ds, epochs=epochs)
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
print("\n begin testing")

pretrained_model_without_top_layer.trainable = True

model.compile(
  optimizer=keras.optimizers.Adam(learning_rate=0.00001),
  loss=losses.CategoricalCrossentropy(from_logits=True),
  metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")]
)

new_history = model.fit(train_ds, validation_data=val_ds, epochs=epochs//5)

accuracy = model.evaluate(X_test, y_test)
print("accuracy: " + str(accuracy))

# plot

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,2.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# create predictions

test_dir = "Data3/test/"
test_data = pd.read_csv("Data3/test/test.csv")

test_files = test_data['Image_Path']
test_imgs =  []

print("being loading testing pictures")
for i in range(len(test_files)):
  print(i)
  img = PIL.Image.open(test_dir + test_files.iloc[i])
  img = img.resize((img_width,img_height))
  test_imgs.append(utils.img_to_array(img))
test_imgs = np.array(test_imgs)
test_imgs /= 255
classes = []

for i in range(len(test_files)):
  prediction = model.predict(np.expand_dims(keras.utils.img_to_array(test_imgs[i],dtype=float),axis = 0))
  classes.append(np.argmax(prediction)+1)
  print(prediction)

submission = pd.DataFrame(
  {"ID":test_files,"Predicted_Classes":classes}
)

loc = SECRET + "trns" + str(epochs) + "+" + str(epochs//5) + "eAug" + ".csv"
submission.to_csv(loc, index=False)