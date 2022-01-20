import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from tensorflow.keras.layers import MaxPool2D,Conv2D,UpSampling2D,Input,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from tensorflow.keras import layers

def down(filters , kernel_size, apply_batch_normalization = True):
    downsample = tf.keras.models.Sequential()
    downsample.add(layers.Conv2D(filters,kernel_size,padding = 'same', strides = 2))
    if apply_batch_normalization:
        downsample.add(layers.BatchNormalization())
    downsample.add(keras.layers.LeakyReLU())
    return downsample


def up(filters, kernel_size, dropout = False):
    upsample = tf.keras.models.Sequential()
    upsample.add(layers.Conv2DTranspose(filters, kernel_size,padding = 'same', strides = 2))
    if dropout:
        upsample.dropout(0.2)
    upsample.add(keras.layers.LeakyReLU())
    return upsample


def model():
    inputs = layers.Input(shape=[160, 160, 3])
    d1 = down(128, (3, 3), False)(inputs)
    d2 = down(128, (3, 3), False)(d1)
    d3 = down(256, (3, 3), True)(d2)
    d4 = down(512, (3, 3), True)(d3)

    d5 = down(512, (3, 3), True)(d4)
    # upsampling
    u1 = up(512, (3, 3), False)(d5)
    u1 = layers.concatenate([u1, d4])
    u2 = up(256, (3, 3), False)(u1)
    u2 = layers.concatenate([u2, d3])
    u3 = up(128, (3, 3), False)(u2)
    u3 = layers.concatenate([u3, d2])
    u4 = up(128, (3, 3), False)(u3)
    u4 = layers.concatenate([u4, d1])
    u5 = up(3, (3, 3), False)(u4)
    u5 = layers.concatenate([u5, inputs])
    output = layers.Conv2D(3, (2, 2), strides=1, padding='same')(u5)
    return tf.keras.Model(inputs=inputs, outputs=output)


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


if __name__ == "__main__":
    # to get the files in proper order


    # defining the size of the image
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
            # open cv reads images in BGR format so we have to convert it to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # resizing image
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

            # resizing image
            img = cv2.resize(img, (SIZE, SIZE))
            img = img.astype('float32') / 255.0
            gray_img.append(img_to_array(img))

    train_gray_image = gray_img[:5500]
    train_color_image = color_img[:5500]

    test_gray_image = gray_img[5500:]
    test_color_image = color_img[5500:]
    # reshaping
    train_g = np.reshape(train_gray_image, (len(train_gray_image), SIZE, SIZE, 3))
    train_c = np.reshape(train_color_image, (len(train_color_image), SIZE, SIZE, 3))
    print('Train color image shape:', train_c.shape)

    test_gray_image = np.reshape(test_gray_image, (len(test_gray_image), SIZE, SIZE, 3))
    test_color_image = np.reshape(test_color_image, (len(test_color_image), SIZE, SIZE, 3))
    print('Test color image shape', test_color_image.shape)

    model = model()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error', metrics=['acc'])

    model.fit(train_g, train_c, epochs=50, batch_size=50)
    for i in range(50, 58):
        predicted = np.clip(model.predict(test_gray_image[i].reshape(1, SIZE, SIZE, 3)), 0.0, 1.0).reshape(SIZE, SIZE, 3)
        plot_images(test_color_image[i], test_gray_image[i], predicted)
