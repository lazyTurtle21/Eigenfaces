from eigenfaces import Eigenfaces
from image import Image
from dataset import Dataset


if __name__ == "__main__":
    # load images
    eigen = Eigenfaces("./orl_faces")
    # # get the weights for single image
    # weights1 = eigen.calculate_weight(eigen.images[0])
    # # show original image & image in reduced basis
    # Image.show_image(eigen.images[0] + eigen.average)
    # Image.show_image(eigen.reverse_image(weights1))

    from scipy.stats import norm

    # creating a class-container for known individual

    eigen.dataset.persons = [*filter(lambda p: p.name != "s2_test", eigen.dataset.persons)]

    recognize_func = "dist"

    # recognizing procedure
    probs = eigen.recognize(Image.read_image("./orl_faces/s2_test/8.pgm"), recognize_func)
    print(probs)
    probs = eigen.recognize(Image.read_image("./orl_faces/s35/6.pgm"), recognize_func)
    print(probs)
    probs = eigen.recognize(Image.read_image("./orl_faces/s24/3.pgm"), recognize_func)
    print(probs)
