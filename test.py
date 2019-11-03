#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model, load_model
import argparse
import matplotlib.pyplot as plt
import numpy as np
from visualize_activations import VisualizeActivation
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2



def test(model, args):
    img = plt.imread(args.image)
    img = cv2.resize(img, (args.image_size, args.image_size))
    X_test = np.reshape(img, (1, img.shape[0],  img.shape[1], img.shape[2]))
    y_pred = model.predict(X_test)

    return y_pred, X_test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', help='model', dest='model', type=str, required=True)
    parser.add_argument('-i', help='image', dest='image', type=str, required=True)
    parser.add_argument('-imsize', help='image size', dest='image_size', type=int, default=128)
    
    args = parser.parse_args()
    print(args)

    model = load_model(args.model)
    vis_obj = VisualizeActivation(17,model)
    y_pred, img = test(model, args)
    vis_obj.visualize_feature_maps(img)

if __name__ == '__main__':
    main()

