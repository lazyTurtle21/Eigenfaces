from processing import Eigenfaces, Image
import cv2

FOLDER = "./normalized_apps"


def test_person(eigen, file):
    probs = eigen.recognize(Image.read_image(FOLDER + "/" + file))
    name = max(probs, key=lambda x: x[1], default=["Failed"])[0]
    print("Detecting %s:" % file)
    print("\tMax:", name)
    print("\tProbabilities:", probs)
    print()

    return name == file.split("/")[0]


def test_accuracy(eigen):
    import os

    true_values = 0
    all_values = 0

    for person_directory in os.listdir(FOLDER):
        path = FOLDER + "/" + person_directory

        if os.path.isdir(path):
            for image in os.listdir(path):
                image_path = path + "/" + image

                image = Image.read_image(image_path)
                result = eigen.recognize(image)

                all_values += 1
                if len(result) > 0 and result[0][0] == person_directory:
                    true_values += 1

    print("Recognized: %d / %d (%f%%)" %
          (true_values, all_values, true_values / all_values * 100))


def test_apps():
    # load images
    eigen = Eigenfaces(FOLDER)

    recognize_func = "norm"

    # recognizing procedure
    # test_person(eigen, "Veronika_Romanko/4.jpg")
    # test_person(eigen, "Yulianna_Tymchenko/7.jpg")
    test_person(eigen, "Mariia_Kulyk/4.jpg")
    test_person(eigen, "Andriy_Dmytruk/4.jpg")
    test_person(eigen, "Andriy_Dmytruk_New/1557603255750815.png")

    test_accuracy(eigen)


def normalize_images(max=100):
    dir_from = "./resized_apps"
    dir_to = "./normalized_apps"

    omit = ["Oleksandr_Sysonov", "Mykola_Biliaev"]

    import os

    all_people = os.listdir(dir_from)
    all_people = [*filter(lambda x: x not in omit, all_people)][:max]

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

    # normalize_images(10)
    test_apps()
