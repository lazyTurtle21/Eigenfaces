import os
import numpy as np


from processing.image import Image


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

        # print([*map(lambda x: x.shape, images)])

        self.images = np.array(images)

    def distance(self, other_weight):
        return np.linalg.norm(self.class_weight - other_weight)

    def distances(self, other_weight):
        return [*map(lambda w: self.distance(other_weight), self.weights)]
