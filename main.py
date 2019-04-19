from eigenfaces import Eigenfaces
from image import Image
from dataset import Dataset


if __name__ == "__main__":
    # load images
    eigen = Eigenfaces("./orl_faces")
    # get the weights for single image
    weights1 = eigen.calculate_weight(eigen.images[0])
    # show original image & image in reduced basis
    Image.show_image(eigen.images[0] + eigen.average)
    Image.show_image(eigen.reverse_image(weights1))

    # creating a class-container for known individuals
    database = Dataset("./orl_faces")

    # recognizing procedure
    name = database.recognize(Image.read_image("./orl_faces/s12/3.pgm"))
    print(name)
    name = database.recognize(Image.read_image("./orl_faces/s35/6.pgm"))
    print(name)
    name = database.recognize(Image.read_image("./orl_faces/s24/3.pgm"))
    print(name)
