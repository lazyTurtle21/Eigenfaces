from processing import Eigenfaces, Image


FOLDER = "./normalized_apps"


def test_person(eigen, file):
    probs = eigen.recognize(Image.read_image(FOLDER + "/" + file))
    name = max(probs, key=lambda x: x[1], default=["Failed"])[0]
    print("Detecting %s:" % file)
    print("\tMax:", name)
    print("\tProbabilities:", probs)
    print()

    return name == file.split("/")[0]


def test_apps():
    # load images
    eigen = Eigenfaces(FOLDER)

    recognize_func = "norm"

    # recognizing procedure
    test_person(eigen, "Veronika_Romanko/4.jpg")
    test_person(eigen, "Yulianna_Tymchenko/7.jpg")
    test_person(eigen, "Mariia_Kulyk/4.jpg")
    test_person(eigen, "Andriy_Dmytruk/4.jpg")


def normalize_images():
    dir_from = "./resized_apps"
    dir_to = "./normalized_apps"

    import os

    for person_directory in os.listdir(dir_from):
        path_from = dir_from + "/" + person_directory
        path_to = dir_to + "/" + person_directory

        if not os.path.isdir(path_to):
            os.makedirs(path_to)

        if os.path.isdir(path_from):
            for image in os.listdir(path_from):
                image_from = path_from + "/" + image
                image_to = path_to + "/" + image

                if os.path.exists(image_to):
                    os.remove(image_to)

                image = Image.find_face(image_from)
                if image is not None:
                    Image.save_image(image, image_to)

    #end :-)


if __name__ == "__main__":
    # normalize_images()
    test_apps()
