import numpy as np
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
from skimage import filters, morphology
from scipy import ndimage as ndi
import cv2
import os


class Image:
    # dict with key - flattened shape, value - original image shape
    shapes = {}

    @staticmethod
    def inside_oval(x, y, size):
        s = size
        return ((x - s / 2) * 1.5) ** 2 + (y - s / 2) ** 2 < (s / 2) ** 2

    @classmethod
    def gamma_correct(cls, image, gamma=2.):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        image = cv2.LUT(image, table)
        return image

    @classmethod
    def dog_correction(cls, image, var1=2, var2=1):
        blur1 = cv2.GaussianBlur(image, (5, 5), var1).astype("float")
        blur2 = cv2.GaussianBlur(image, (5, 5), var2).astype("float")

        dog = blur1 - blur2
        dog = dog - min(dog.flatten()) / (max(dog.flatten()) * 255)
        return dog.astype("uint8")

    @classmethod
    def histogram_equalization(cls, image):
        L = 256
        pdf = [0 for i in range(L)]

        for x, y in np.ndindex(image.shape):
            pdf[image[x, y]] += 1

        pdf = [*map(lambda x: x / (image.shape[0] * image.shape[1]), pdf)]

        cdf = [0 for i in range(L)]
        cdf[0] = pdf[0]
        for i in range(1, L):
            cdf[i] = cdf[i - 1] + pdf[i]

        table = np.floor(np.array(cdf) * L).astype("uint8")
        image = cv2.LUT(image, table)
        return image

    @classmethod
    def cut_oval(cls, image):
        size = image.shape[1]

        for x, y in np.ndindex(image.shape):
            image[x, y] = image[x, y] if Image.inside_oval(x, y, size) else 0

        return image

    @classmethod
    def find_face(cls, filename):
        SCALE = (0.63, 0.9)
        SIZE = (150, 200)

        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        current_path = os.path.dirname(os.path.abspath(__file__))
        cascade_name = current_path + "/haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_name)

        faces = face_cascade.detectMultiScale(
            image,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return None

        (x, y, w, h) = faces[0]
        x = int(x - (SCALE[0] - 1) / 2 * w)
        y = int(y - (SCALE[1] - 1) / 2 * h)
        w = int(SCALE[0] * w)
        h = int(SCALE[1] * h)

        image = image[y:y+h, x:x+w]

        image = cv2.resize(image, dsize=SIZE,
                           interpolation=cv2.INTER_CUBIC)

        image = Image.histogram_equalization(image)
        image = Image.gamma_correct(image, 1.8)


        # image = Image.dog_correction(image, 8, 1)

        g_kernel = cv2.getGaborKernel((5, 8), 2 * np.pi, np.pi / 2, 9.8, 5 ** 2, 0, ktype=cv2.CV_32F)
        image = cv2.filter2D(image, cv2.CV_8UC3, g_kernel)

        # image = Image.cut_oval(image)

        image = image.astype("float")
        image /= max(image.flatten())

        # Image.show_image(image)

        return image

    @classmethod
    def remove_background(cls, image):
        size = image.shape[0]

        def inside_oval(x, y):
            s = size
            return ((x - s / 2) * 1.5) ** 2 + (y - s / 2) ** 2 < (s / 2) ** 2

        edges = filters.sobel(image)
        edges = filters.gaussian(edges, sigma=1.5)
        np.multiply(edges, 255 / max(edges.flatten()), out=edges, casting="unsafe")

        plt.imshow(edges)
        plt.show()

        light_spots = np.array((image > 250).nonzero()).T
        dark_spots = np.array((image < 5).nonzero()).T

        plt.plot(dark_spots[:, 1], dark_spots[:, 0], 'o')
        plt.imshow(image)
        plt.title('light spots in image')
        plt.show()

        bool_mask = np.zeros(image.shape, dtype=np.bool)
        bool_mask[tuple(light_spots.T)] = True
        bool_mask[tuple(dark_spots.T)] = True
        seed_mask, num_seeds = ndi.label(bool_mask)

        ws = morphology.watershed(edges, seed_mask)

        plt.imshow(ws)
        plt.show()

        groups = set(ws.ravel())
        for group in groups:
            print(group)
            outside = False

            for x, y in np.ndindex(ws.shape):
                if ws[x, y] == group and not inside_oval(x, y):
                    outside = True

            if not outside: continue

            for x, y in np.ndindex(ws.shape):
                if ws[x, y] == group:
                    image[x, y] = 0

        # background = max(set(ws.ravel()), key=lambda g: np.sum(ws == g))
        # background_mask = (ws == background)
        # cleaned = image * ~background_mask

        return image

    @classmethod
    def read_image(cls, path):
        image = cls.read_image_2d(path)

        cls.shapes[image.shape[0] * image.shape[1]] = image.shape
        image = image.flatten()
        return image

    @classmethod
    def read_image_2d(cls, path):
        # image = mpimage.imread(path)
        #
        # if len(image.shape) == 3 and image.shape[2] >= 3:
        #     r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        #     image = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # elif len(image.shape) != 2:
        #     raise Exception("Invalid image dimensions: " + str(image.shape))
        #
        # image = image / max(image.flatten())

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype("float")
        image /= max(image.flatten())

        return image

    @classmethod
    def save_image(cls, image, path):
        mpimage.imsave(path, image, cmap=plt.get_cmap("gray"))

    @classmethod
    def show_image(cls, image, title=""):
        # get original shape
        if len(image.shape) == 1:
            image = np.reshape(image, cls.get_image_shape(image))

        # display the image
        plt.title(title)
        plt.imshow(image, interpolation='nearest', cmap=plt.get_cmap("gray"))
        plt.show()

    @classmethod
    def get_image_shape(cls, image):
        return cls.shapes[image.shape[0]]
