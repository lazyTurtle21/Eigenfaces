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


def get_image_paths():
    paths = []
    import os

    for person_directory in os.listdir(FOLDER):
        path = FOLDER + "/" + person_directory

        if os.path.isdir(path):
            for image in os.listdir(path):
                image_path = path + "/" + image

                paths.append((image_path, person_directory))

    return paths


def test_accuracy_good():
    import random
    import os

    test_frac = 0.10

    paths = get_image_paths()
    true_values = 0

    random.shuffle(paths)
    test_paths = paths[:int(len(paths) * test_frac)]
    images = []

    for test_path, name in test_paths:
        images.append(Image.read_image_2d(test_path))
        os.remove(test_path)

    eigen = Eigenfaces(FOLDER)
    for i in range(len(test_paths)):
        result = eigen.recognize(images[i].flatten())
        result = result[0][0] if len(result) > 0 else None

        if result == test_paths[i][1]:
            true_values += 1

    for i in range(len(images)):
        Image.save_image(images[i], test_paths[i][0])

    print("[Eigenfaces] Real accuracy: %d / %d (%d%%)" %
          (true_values, len(test_paths), true_values / len(test_paths) * 100))

    return true_values / len(test_paths)


def test_accuracy(eigen):
    true_values = 0
    all_values = 0

    for image_path, name in get_image_paths():
        image = Image.read_image(image_path)
        result = eigen.recognize(image)

        all_values += 1
        if len(result) > 0 and result[0][0] == name:
            true_values += 1

    print("[Eigenfaces] Accuracy: %d / %d (%f%%)" %
          (true_values, all_values, true_values / all_values * 100))


def test_apps():
    # load images
    eigen = Eigenfaces(FOLDER)

    recognize_func = "norm"

    # recognizing procedure
    # test_person(eigen, "Veronika_Romanko/4.jpg")
    # test_person(eigen, "Yulianna_Tymchenko/7.jpg")
    # test_person(eigen, "Mariia_Kulyk/4.jpg")
    # test_person(eigen, "Andriy_Dmytruk/4.jpg")
    # test_person(eigen, "Andriy_Dmytruk_New/1557603255750815.png")

    test_accuracy(eigen)

    accuracy = 0
    num = 10
    for i in range(num):
        accuracy += test_accuracy_good()

    print("[Eigenfaces] Average real accuracy: %f%%" % (accuracy / num * 100))


def normalize_images(max=100):
    dir_from = "./apps_faces"
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

    normalize_images()
    test_apps()
