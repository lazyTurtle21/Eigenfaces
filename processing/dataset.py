import os
import numpy as np
from scipy.stats import norm as stats_norm

from processing.person import Person


class Dataset:
    def __init__(self, directory):
        self.persons = None
        self.all_images = None
        self.initialize(directory)

        self.average_dist = None
        self.standard_deviation = None

    def normalize_people(self):
        self.persons = [*filter(lambda x: len(x.images) > 0, self.persons)]

        number_images = len(self.all_images)
        if number_images > 0:
            shape = self.all_images[0].shape

            def filter_shape(images):
                return np.array([*filter(lambda i: i.shape == shape, images)])

            self.all_images = filter_shape(self.all_images)
            for person in self.persons:
                person.images = filter_shape(person.images)

        if len(self.all_images) != number_images:
            print("[Eigenfaces Warning] %d images did not match dimensions" %
                  (number_images - len(self.all_images)))

    def initialize(self, directory):
        persons = []
        all_images = []

        for person_directory in os.listdir(directory):
            person_directory_path = directory + "/" + person_directory
            if os.path.isdir(person_directory_path):
                person = Person(person_directory_path)

                persons.append(person)
                all_images.extend(person.images)

        self.persons = np.array(persons)
        self.all_images = np.array(all_images)

        self.normalize_people()

        print("[Eigenfaces] Images loaded", self.all_images.shape[0])

    def calculate_weights(self, weight_func):
        for person in self.persons:
            person.weights = np.apply_along_axis(weight_func, 1, person.images)
            person.class_weight = np.apply_along_axis(np.mean, 0,
                                                      person.weights)

    def __recognize_with_probabilities__(self, probabilities):
        # probabilities for each person in the order of self.persons

        indexes = np.flip(np.argsort(probabilities), 0)

        # filter the largest probabilities
        indexes = [*filter(lambda i: 4 * probabilities[i] >
                           probabilities[indexes[0]], indexes)]

        # return (name: probability) pairs to user
        return [*map(lambda i: (self.persons[i].name, probabilities[i]),
                     indexes)]

    def initialize_norm(self):
        distances = []

        for person in self.persons:
            person_distances = np.apply_along_axis(np.linalg.norm, 1,
                                                   person.class_weight
                                                   - person.weights)
            distances.extend(person_distances)

        distances = np.array(distances)
        self.average_dist = np.average(distances)
        self.standard_deviation = np.std(distances)

    def recognize_norm(self, weight):
        # recognize using normal distribution
        if not self.average_dist:
            self.initialize_norm()

        distances = np.array([*map(lambda x: x.distance(weight), self.persons)])

        # create the normal distribution and get the probabilities
        normal = stats_norm(self.average_dist, self.standard_deviation)
        probabilities = np.vectorize(lambda x: 1 - normal.cdf(x))(distances)

        return self.__recognize_with_probabilities__(probabilities)

    def recognize_dist(self, weight):
        # recognize by the distances fraction
        distances_inv = np.array([*map(lambda x: 1 / x.distance(weight),
                                       self.persons)])
        distance_isum = np.sum(distances_inv)

        probabilities = distances_inv / distance_isum

        return self.__recognize_with_probabilities__(probabilities)

    def __iter__(self):
        for person in self.persons:
            yield person

