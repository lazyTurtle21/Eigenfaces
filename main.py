from eigenfaces import Eigenfaces
from image import Image


if __name__ == "__main__":
    # load images
    eigen = Eigenfaces("./resized_apps")
    # # get the weights for single image
    # weights1 = eigen.calculate_weight(eigen.images[0])
    # # show original image & image in reduced basis
    # Image.show_image(eigen.images[0] + eigen.average)
    # Image.show_image(eigen.reverse_image(weights1))

    # creating a class-container for known individual

    # eigen.dataset.persons = [*filter(lambda p: p.name != "s2_test",
    #                                  eigen.dataset.persons)]

    recognize_func = "dist"

    # recognizing procedure
    probs = eigen.recognize(Image.read_image(
        "./resized_apps/Veronika_Romanko/4.jpg"),
                            recognize_func)
    print(max(probs, key=lambda x: x[1]))
    print(probs)
    probs = eigen.recognize(Image.read_image("./resized_apps/Yulianna_Tymchenko/7.jpg"),
                            recognize_func)
    print(max(probs, key=lambda x: x[1]))
    print(probs)
    probs = eigen.recognize(Image.read_image("./resized_apps/Mariia_Kulyk/4.jpg"),
                            recognize_func)
    print(max(probs, key=lambda x: x[1]))
    print(probs)
