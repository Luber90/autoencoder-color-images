import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from tensorflow.keras.layers import Conv2D,Dropout,BatchNormalization,Conv2DTranspose
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import img_to_array
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import pickle



class AutoEncoder(keras.models.Model):
    def __init__(self, dropout_rate = 0.2, momentum = 0.99, decay = 0.0):
        super(AutoEncoder, self).__init__(name='autoencoder')
        self.layer1 = Conv2D(128, (5, 5), padding='same', strides=2, activation="relu", input_shape=(160, 160, 3))
        self.layer2 = Conv2D(128, (5, 5), padding='same', strides=2, activation="relu")
        self.layer3 = Conv2D(256, (4, 4), padding='same', strides=2, activation="relu")
        self.layer4 = BatchNormalization(momentum=momentum)
        self.layer5 = Conv2D(512, (3, 3), padding='same', strides=2, activation="relu", bias_regularizer=l2(decay), kernel_regularizer=l2(decay),activity_regularizer=l2(decay))
        self.layer6 = BatchNormalization(momentum=momentum)
        self.layer7 = Conv2D(512, (3, 3), padding='same', strides=2, activation="relu")
        self.layer8 = BatchNormalization(momentum=momentum)
        self.layer9 = Conv2DTranspose(512, (3, 3), padding='same', strides=2, activation="relu")
        self.layer10 = Dropout(dropout_rate)
        self.layer11 = Conv2DTranspose(512, (3, 3), padding='same', strides=2, activation="relu", bias_regularizer=l2(decay), kernel_regularizer=l2(decay),activity_regularizer=l2(decay))
        self.layer12 = Conv2DTranspose(256, (4, 4), padding='same', strides=2, activation="relu")
        self.layer13 = Conv2DTranspose(128, (5, 5), padding='same', strides=2, activation="relu")
        self.layer14 = Conv2DTranspose(128, (5, 5), padding='same', strides=2, activation="relu")
        self.layer15 = Conv2D(3, (2, 2), padding='same', strides=1)

    def call(self, inputs):
        x1 = self.layer1(inputs)
        x2 = self.layer2(x1)
        x = self.layer3(x2)
        x4 = self.layer4(x)
        x = self.layer5(x4)
        x6 = self.layer6(x)
        x = self.layer7(x6)
        x = self.layer8(x)

        x = self.layer9(x)
        x = self.layer10(x)
        x = layers.concatenate([x, x6])
        x = self.layer11(x)
        x = layers.concatenate([x, x4])
        x = self.layer12(x)
        x = layers.concatenate([x, x2])
        x = self.layer13(x)
        x = layers.concatenate([x, x1])
        x = self.layer14(x)
        x = layers.concatenate([x, inputs])
        x = self.layer15(x)

        return x


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

    train_gray_image = gray_img[:5500]
    train_color_image = color_img[:5500]

    test_gray_image = gray_img[5500:6000]
    test_color_image = color_img[5500:6000]

    # reshaping
    train_g = np.reshape(train_gray_image, (len(train_gray_image), SIZE, SIZE, 3))
    train_c = np.reshape(train_color_image, (len(train_color_image), SIZE, SIZE, 3))
    print('Train color image shape:', train_c.shape)

    test_gray_image = np.reshape(test_gray_image, (len(test_gray_image), SIZE, SIZE, 3))
    test_color_image = np.reshape(test_color_image, (len(test_color_image), SIZE, SIZE, 3))
    print('Test color image shape', test_color_image.shape)

    epochy = 50
    folder = "decay/"
    name = "model3"
    model = AutoEncoder(decay=0.1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error', metrics=['acc'])
    history = model.fit(train_g, train_c, epochs=epochy, batch_size=30, validation_data=[test_gray_image, test_color_image])

    with open(folder + name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)



    '''
    batch momentum
        model1 batch=15 momentum 0.99
        model2 batch=30 momentum 0.99
        model3 batch 30 momentum 0.95
        
    dropout
        model1 dropout 0
        model2 dropout 0.2
        model3 dropout 0.5
    
    decay
        model1 decay 0
        model2 0.001
        model3 0.01
        model4 0.1
    '''

