import numpy
import matplotlib.image as mpimage


class Image:
    @staticmethod
    def read_image(path):
        rgb = mpimage.imread(path)

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = gray.flatten()

        return gray

