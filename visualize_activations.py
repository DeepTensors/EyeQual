from keras.models import Model, load_model
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
import numpy as np

class VisualizeActivation:
    def __init__(self, layer, model):
        self.layers = layer
        self.model = model

    def compute_feature_maps(self, image):
        model_ = Model(inputs=self.model.inputs, outputs=self.model.layers[self.layers].output)
        return model_.predict(image)

    def visualize_feature_maps(self, image, gray=False):
        #just visualize first 16 channels of the feature map
        i = 1
        feature_maps = self.compute_feature_maps(image)
        for _ in range(1):
                ax = plt.subplot(1, 1, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                if gray:
                    plt.imshow(feature_maps[0, :, :, i - 1],cmap='gray')
                else:
                    plt.imshow(feature_maps[0, :, :, i - 1])

        plt.show()


