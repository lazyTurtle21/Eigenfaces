import os
import numpy as np

from image import Image


class Person:
    def __init__(self, path):
        self.name = path.split("/")[-1]
        self.path = path
        self.number_pic = 0
        self.class_vector = None

    def initialize(self, eigenfaces):
        weights = []

        # calculating weights for all images of given person
        if os.path.isdir(self.path):
            for image in os.listdir(self.path):
                image = Image.read_image(self.path + "/" + image) \
                        - eigenfaces.average
                weights.append(eigenfaces.calculate_weight(image))
        weights = np.array(weights)

        # take the average over weights
        self.class_vector = np.mean(weights, axis=0)
        self.number_pic = weights.shape[0]

    def distance(self, person2):
        return np.linalg.norm(self.class_vector - person2)
