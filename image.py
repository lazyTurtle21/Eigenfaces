import numpy as np
import matplotlib.image as mpimage
import matplotlib.pyplot as plt


class Image:
    # dict with key - flattened shape, value - original image shape
    shapes = {}

    @classmethod
    def read_image(cls, path):
        image = cls.read_image_2d(path)

        cls.shapes[image.shape[0] * image.shape[1]] = image.shape
        image = image.flatten()
        return image

    @classmethod
    def read_image_2d(cls, path):
        image = mpimage.imread(path)

        if len(image.shape) == 3 and image.shape[2] >= 3:
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            image = 0.2989 * r + 0.5870 * g + 0.1140 * b
        elif len(image.shape) != 2:
            raise Exception("Invalid image dimensions: " + str(image.shape))

        image = image / 255
        return image

    @classmethod
    def show_image(cls, image):
        # get original shape
        if len(image.shape) == 1:
            image = np.reshape(image, cls.get_image_shape(image))

        # display the image
        plt.imshow(image, interpolation='nearest', cmap=plt.get_cmap("gray"))
        plt.show()

    @classmethod
    def get_image_shape(cls, image):
        return cls.shapes[image.shape[0]]
