from eigenfaces import Eigenfaces
from image import Image


if __name__ == "__main__":
    # load images
    eigenfaces = Eigenfaces("./orl_faces")
    # get the weights for single image
    weights1 = eigenfaces.calculate_weight(eigenfaces.images[0])

    # show original image & image in reduced basis
    Image.show_image(eigenfaces.images[0] + eigenfaces.average)
    Image.show_image(eigenfaces.reverse_image(weights1))
