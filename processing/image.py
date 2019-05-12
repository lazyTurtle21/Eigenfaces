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

    @classmethod
    def find_face(cls, filename):
        SCALE = 1.2
        SIZE = 100

        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        current_path = os.path.dirname(os.path.abspath(__file__))
        cascade_name = current_path + "/haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_name)

        faces = face_cascade.detectMultiScale(
            image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # for (x, y, w, h) in faces:
        #     cv2.rectangle(image, (x, y), (x + w, y + h), 0, 2)
        # Image.show_image(image)

        if len(faces) == 0:
            return None

        (x, y, w, h) = faces[0]
        x = int(x - (SCALE - 1) / 2 * w)
        y = int(y - (SCALE - 1) / 2 * h)
        w = int(SCALE * w)
        h = int(SCALE * h)

        image = image[y:y+h, x:x+w]

        image = cv2.resize(image, dsize=(SIZE, SIZE),
                           interpolation=cv2.INTER_CUBIC)

        def inside_oval(x, y):
            s = SIZE
            return ((x - s / 2) * 1.5) ** 2 + (y - s / 2) ** 2 < (s / 2) ** 2

        for x, y in np.ndindex(image.shape):
            image[y, x] = image[y, x] if inside_oval(x, y) else 0

        return image

    @classmethod
    def remove_background(cls, image):
        size = image.shape[0]

        def inside_oval(x, y):
            s = size
            return ((x - s / 2) * 1.5) ** 2 + (y - s / 2) ** 2 < (s / 2) ** 2

        edges = filters.sobel(image)
        edges = filters.gaussian(edges, sigma=1.8)
        np.multiply(edges, 255 / max(edges.flatten()), out=edges, casting="unsafe")

        plt.imshow(edges)
        plt.show()

        light_spots = np.array((image > 240).nonzero()).T
        dark_spots = np.array((image < 20).nonzero()).T

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
                if ws[x, y] == group and inside_oval(x, y):
                    outside = True

            if outside: continue

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
        image = mpimage.imread(path)

        if len(image.shape) == 3 and image.shape[2] >= 3:
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            image = 0.2989 * r + 0.5870 * g + 0.1140 * b
        elif len(image.shape) != 2:
            raise Exception("Invalid image dimensions: " + str(image.shape))

        image = image / max(image.flatten())

        # image = cv2.imread(path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # np.multiply(image, 255 / max(image.flatten()),
        #             out=image, casting="unsafe")

        return image


    @classmethod
    def save_image(cls, image, path):
        mpimage.imsave(path, image, cmap=plt.get_cmap("gray"))

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
