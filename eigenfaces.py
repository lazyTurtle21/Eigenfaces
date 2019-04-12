import os
import numpy as np
from image import Image


class Eigenfaces:
    def __init__(self, directory):
        images = []

        for filename in os.listdir(directory):
            images.append(Image.read_image(directory + "/" + filename))

        images = np.array(images)

        self.average = self.get_average(images)
        self.images = self.normalize_all(images)

        print(self.images.shape)

    @staticmethod
    def get_average(images):
        return np.sum(images, axis=0) / images.shape[0]

    def normalize_all(self, images):
        average = self.average
        return np.apply_over_axes(lambda x, i: x - average, images, [1])


if __name__ == "__main__":
    eigenfaces = Eigenfaces("./images")
