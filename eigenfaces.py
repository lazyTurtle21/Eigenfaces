import numpy as np

from dataset import Dataset


class Eigenfaces:
    def __init__(self, directory, threshold=0.8):
        self.dataset = Dataset(directory)

        self.threshold = threshold
        self.average = self.get_average(self.dataset.all_images)
        self.vectors = None
        self.k = None
        self.vectorsT = None

        self.create_reduced_basis()
        self.dataset.calculate_weights(self.calculate_weight)

    @staticmethod
    def get_average(images):
        return np.sum(images, axis=0) / images.shape[0]

    def create_reduced_basis(self):
        images = self.dataset.all_images
        # normalize all the images - subtract the average
        images = np.apply_along_axis(lambda image: image - self.average, 1, images)

        # covariance matrix of images
        covariance = np.matmul(images, np.matrix.transpose(images))

        # find eigenvectors for image basis
        # (image_size x num_vectors)
        values, vectors = np.linalg.eig(covariance)
        vectors = np.matmul(np.transpose(images), vectors)

        # get number of principal components
        indexes = np.flip(np.argsort(values), 0)
        values = values[indexes]
        variance = np.sum(values)
        length = values.size
        threshold_var = 0
        k = 0

        # choose k that covers (threshold * 100%) percent of the variance
        while threshold_var / variance < self.threshold and k < length:
            threshold_var += values[k]
            k += 1

        print("K is", k)
        self.k = k
        vectors = vectors[:, indexes[0:self.k]]

        # make the basis orthonormal
        self.vectors = self.make_orthogonal(vectors)
        self.vectorsT = np.transpose(self.vectors)

    @staticmethod
    def make_orthogonal(vectors):
        new_vectors = []

        for i in range(vectors.shape[1]):
            vector = vectors[:, i]

            for j in range(i):
                basis_v = vectors[:, j]
                projection = basis_v * \
                    np.dot(basis_v, vector) / np.dot(basis_v, basis_v)
                vector = vector - projection

            new_vectors.append(vector)

        new_vectors = np.array(new_vectors)
        new_vectors = np.apply_along_axis(
            lambda x: x / np.sqrt(np.dot(x, x)), 1, new_vectors)

        return np.transpose(new_vectors)

    def calculate_weight(self, image):
        # calculate the weight of image - in the reduced basis
        # needs to be subtracted the average first
        return np.matmul(image - self.average, self.vectors)

    def reverse_image(self, weights):
        # return image back to normal
        return np.matmul(weights, self.vectorsT) + self.average

    def recognize(self, image, func="norm"):
        # calculate the weight
        weight = self.calculate_weight(image)

        if func == "norm":
            return self.dataset.recognize_norm(weight)
        elif func == "dist":
            return self.dataset.recognize_dist(weight)

