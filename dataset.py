import os

from person import Person
from eigenfaces import Eigenfaces


#   TO DO: refactor

class Dataset:
    def __init__(self, directory):
        self.persons = []
        self.eigenfaces = Eigenfaces(directory)
        self.initialize(directory)

    def initialize(self, directory):
        for person_directory in os.listdir(directory):
            person_directory_path = directory + "/" + person_directory
            if os.path.isdir(person_directory_path):
                person = Person(person_directory_path)
                self.persons.append(person)
                person.initialize(self.eigenfaces)

    def recognize(self, image):
        # threshold to be estimated
        threshold = 10

        image = image - self.eigenfaces.average
        unknown = self.eigenfaces.calculate_weight(image)

        # finding the closest person in dataset
        d, person = min(list(zip(map(lambda x: x.distance(unknown),
                                     self.persons), self.persons)))
        if d < threshold:
            return "Person's name: " + person.name
        else:
            return "No such person"

    def __iter__(self):
        for person in self.persons:
            yield person
