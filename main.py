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
    test_person(eigen, "Andriy_Dmytruk_New/1557603255750815.png")


def normalize_images():
    dir_from = "./resized_apps"
    dir_to = "./normalized_apps"

    import os

    all_people = [*os.listdir(dir_from)]
    for i in range(len(all_people)):
        person_directory = all_people[i]
        print("[Eigenfaces] Normalizing person %d / %d" % (i + 1, len(all_people)))

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

    # end :-)


def test_detection(file):
    image = Image.find_face(file)
    print(image)
    Image.show_image(image)


if __name__ == "__main__":
    # test_detection("resized_apps/Andriy_Dmytruk_New/1557603255750815.png")
    # test_detection("resized_apps/Andriy_Dmytruk/1.jpg")

    normalize_images()
    test_apps()
