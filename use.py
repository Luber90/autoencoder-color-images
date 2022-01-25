import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from tensorflow.keras.layers import MaxPool2D,Conv2D,UpSampling2D,Input,Dropout,BatchNormalization,Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from tensorflow.keras import layers

def plot_images(color, grayscale, predicted):
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.title('Color Image', color='green', fontsize=20)
    plt.imshow(color)
    plt.subplot(1, 3, 2)
    plt.title('Grayscale Image ', color='black', fontsize=20)
    plt.imshow(grayscale)
    plt.subplot(1, 3, 3)
    plt.title('Predicted Image ', color='Red', fontsize=20)
    plt.imshow(predicted)

    plt.show()



def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

SIZE = 160
color_img = []
path = 'color'
files = os.listdir(path)
files = sorted_alphanumeric(files)
for i in tqdm(files):
    if i == '6000.jpg':
        break
    else:
        img = cv2.imread(path + '/' + i, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        color_img.append(img_to_array(img))

gray_img = []
path = 'gray'
files = os.listdir(path)
files = sorted_alphanumeric(files)
for i in tqdm(files):
    if i == '6000.jpg':
        break
    else:
        img = cv2.imread(path + '/' + i, 1)

        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        gray_img.append(img_to_array(img))

train_gray_image = gray_img[:450]
train_color_image = color_img[:450]

test_gray_image = gray_img[5500:]
test_color_image = color_img[5500:]
# reshaping
train_g = np.reshape(train_gray_image, (len(train_gray_image), SIZE, SIZE, 3))
train_c = np.reshape(train_color_image, (len(train_color_image), SIZE, SIZE, 3))
print('Train color image shape:', train_c.shape)

test_gray_image = np.reshape(test_gray_image, (len(test_gray_image), SIZE, SIZE, 3))
test_color_image = np.reshape(test_color_image, (len(test_color_image), SIZE, SIZE, 3))
print('Test color image shape', test_color_image.shape)

model = keras.models.load_model('model')
for i in range(50, 58):
    predicted = np.clip(model.predict(test_gray_image[i].reshape(1, SIZE, SIZE, 3)), 0.0, 1.0).reshape(SIZE, SIZE, 3)
    plot_images(test_color_image[i], test_gray_image[i], predicted)