from eigenfaces import Eigenfaces
from image import Image
import os
import numpy as np


class Dataset:
    def __init__(self, directory):
        self.persons = []
        self.initialize(directory)

    def initialize(self, directory):
        for person_directory in os.listdir(directory):
            person_directory_path = directory + "/" + person_directory
            if os.path.isdir(person_directory_path):
                self.persons.append(Person(person_directory_path))

    def recognize(self, image):
        # threshold to be estimated
        threshold = 100

        image = image - eigenfaces.average
        unknown = eigenfaces.calculate_weight(image)
        # finding the closest person in dataset
        d, person = min(list(zip(map(lambda x: x.distance(unknown), self.persons), self.persons)))
        if d < threshold:
            return person.name

    def __iter__(self):
        for person in self.persons:
            yield person


class Person:
    def __init__(self, path):
        self.name = path.split("/")[-1]
        self.number_pic = 0
        self.class_vector = None

        # calculating representing class vector
        self.initialize(path)

    def initialize(self, path):
        weights = []

        # calculating weights for all images of given person
        if os.path.isdir(path):
            for image in os.listdir(path):
                image = Image.read_image(path + "/" + image) - eigenfaces.average
                weights.append(eigenfaces.calculate_weight(image))
        weights = np.array(weights)

        # take the average over weights
        self.class_vector = np.mean(weights, axis=0)
        self.number_pic = weights.shape[0]

    def distance(self, person2):
        return np.linalg.norm(self.class_vector - person2)


if __name__ == "__main__":
    eigenfaces = Eigenfaces("./orl_faces")

    # creating a class-container for known individuals
    database = Dataset("./orl_faces")

    # recognizing procedure
    name = database.recognize(Image.read_image("./orl_faces/s34/4.pgm"))
    print(name)
