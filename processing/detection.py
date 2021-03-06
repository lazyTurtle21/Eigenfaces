from image import Image
from processing import Eigenfaces
from dataset import Dataset
import numpy as np

class FaceDetection:
    def __init__(self, eigenfaces):
        self.eigenfaces = eigenfaces

        self.face_shape = Image.get_image_shape(eigenfaces.average)

    def detect(self, path):
        eigenfaces = self.eigenfaces
        face_shape = self.face_shape

        image = Image.read_image_2d(path)

        x = 117
        y = 230
        r_face = image[y:y+face_shape[0], x:x+face_shape[1]]
        Image.show_image(Image.remove_background(r_face).flatten())
        Image.show_image(Image.read_image("./orl_faces/s2/1.pgm"))
        print(eigenfaces.recognize(Image.remove_background(r_face).flatten()))
        print(eigenfaces.recognize(Image.read_image("./orl_faces/s2/1.pgm")))

        result = []
        for x in range(0, image.shape[1] - face_shape[1], face_shape[1] // 8):
            part_result = []
            print(x)

            for y in range(0, image.shape[0] - face_shape[0], face_shape[0] // 8):
                test_face = image[y:y+face_shape[0], x:x+face_shape[1]]

                test_face = test_face.flatten()
                reduced_test_face = \
                    eigenfaces.calculate_weight(test_face - eigenfaces.average)
                reduced_test_face = eigenfaces.reverse_image(reduced_test_face)

                distance = np.linalg.norm(test_face - reduced_test_face)

                part_result.append(distance)

            result.append(part_result)

        result = np.array(result).T
        Image.show_image(result)


if __name__ == "__main__":
    eigenfaces = Eigenfaces("./orl_faces")
    detection = FaceDetection(eigenfaces)
    detection.detect("./detection/muzhyk2.png")
