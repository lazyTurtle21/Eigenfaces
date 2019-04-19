import os
import numpy as np

from image import Image


class Person:
    def __init__(self, path, eigenfaces):
        self.name = path.split("/")[-1]
        self.number_pic = 0
        self.class_vector = None
        self.eigenfaces = eigenfaces

        # calculating representing class vector
        self.initialize(path)

    def initialize(self, path):
        weights = []

        # calculating weights for all images of given person
        if os.path.isdir(path):
            for image in os.listdir(path):
                image = Image.read_image(path + "/" + image) - self.eigenfaces.average
                weights.append(self.eigenfaces.calculate_weight(image))
        weights = np.array(weights)

        # take the average over weights
        self.class_vector = np.mean(weights, axis=0)
        self.number_pic = weights.shape[0]

    def distance(self, person2):
        return np.linalg.norm(self.class_vector - person2)
