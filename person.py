<<<<<<< HEAD
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
=======
import os
import numpy as np
from scipy.stats import norm

from image import Image


class Person:
    def __init__(self, path):
        self.name = path.split("/")[-1]
        self.path = path
        self.images = None
        self.weights = None
        self.class_weight = None

        self.initialize()

    def initialize(self):
        images = []

        if os.path.isdir(self.path):
            for image in os.listdir(self.path):
                images.append(Image.read_image(self.path + "/" + image))

        self.images = np.array(images)

    def distance(self, other_weight):
        return np.linalg.norm(self.class_weight - other_weight)

>>>>>>> 1f7c8d25cb31c51171a318e89d368313513ed870
