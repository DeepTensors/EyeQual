#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
import argparse
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import os
import glob
from eyeCalc import EyeCalc
from keras.optimizers import Adam



def image_preprocessing(image, img_size):
    img = image
    img = cv2.resize(img, (img_size, img_size))

    return img


def read_imgs(image_path,img_size, good=True):
    image_array = []
    label_array = []
    for image in image_path:
        img = plt.imread(image)
        image_array.append(image_preprocessing(img, img_size))
        if good:
            label_array.append(0)
        else:
            label_array.append(1)

    return image_array, label_array


def prepare_data(img_size):
    cwd = os.getcwd()

    path_good_images = glob.glob(os.path.join(cwd, 'Good/*.jpg'))
    path_bad_images = glob.glob(os.path.join(cwd, 'Bad/*.jpg'))

    images_good, labels_good = read_imgs(path_good_images, img_size)
    images_bad, labels_bad = read_imgs(path_bad_images, img_size, good=False)

    X = images_good + images_bad
    Y = labels_good + labels_bad

    X_final = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], np.shape(X)[2], 3))
    Y_final = np.reshape(Y, (np.shape(Y)[0], 1))

    return X_final, Y_final


def model(img_size, args):
    print(img_size)
    obj = EyeCalc(Input(shape=(img_size, img_size, 3)), args.pooling_type)
    model = obj.build_model()
    model.summary()

    return model


def train(model, X, Y, args):

    optimizer = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                monitor='val_acc', verbose=1, save_best_only=True)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
    datagen = ImageDataGenerator(
	vertical_flip=True,
	horizontal_flip=True,
	rotation_range=20,
	rescale=1/255.0,
    )

    datagen.fit(X)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=args.batch_size), callbacks=[checkpoint], epochs=args.epochs, 
                                     verbose=1, validation_data=datagen.flow(x_test, y_test), validation_steps=args.batch_size, steps_per_epoch=2000, shuffle=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=16)
    parser.add_argument('-e', help='epochs', dest='epochs', type=int,  default=10)
    parser.add_argument('-imsize', help='image size', dest='image_size', type=int, default=128)
    parser.add_argument('--pooling', help='poolingtype(SWAP/WAP)', dest='pooling_type', type=str, default='SWAP')
    args = parser.parse_args()
    
    img_size = args.image_size

    X, Y = prepare_data(img_size)
    print(np.shape(X), np.shape(Y))
    model_ = model(img_size, args)
    train(model_, X, Y, args)


if __name__ == '__main__':
    main()
