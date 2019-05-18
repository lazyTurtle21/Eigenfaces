from processing import Eigenfaces, Image
import sys
import random
import os

def get_image_paths(folder):
    paths = []

    for person_directory in os.listdir(folder):
        path = folder + "/" + person_directory

        if os.path.isdir(path):
            for image in os.listdir(path):
                image_path = path + "/" + image

                paths.append((image_path, person_directory))

    return paths


def test_accuracy_good(folder):
    test_frac = 0.20

    paths = get_image_paths(folder)
    true_values = 0

    random.shuffle(paths)
    test_paths = paths[:int(len(paths) * test_frac)]
    images = []

    for test_path, name in test_paths:
        images.append(Image.read_image_2d(test_path))
        os.remove(test_path)

    eigen = Eigenfaces(folder)
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


def test_accuracy(eigen, folder):
    true_values = 0
    all_values = 0

    for image_path, name in get_image_paths(folder):
        image = Image.read_image(image_path)
        result = eigen.recognize(image)

        all_values += 1
        if len(result) > 0 and result[0][0] == name:
            true_values += 1

    print("[Eigenfaces] Accuracy: %d / %d (%f%%)" %
          (true_values, all_values, true_values / all_values * 100))


def test_all(folder, number=10):
    # load images
    eigen = Eigenfaces(folder)

    test_accuracy(eigen, folder)

    accuracy = 0
    num = 10
    for i in range(num):
        accuracy += test_accuracy_good(folder)

    print("[Eigenfaces] Average real accuracy: %f%%" % (accuracy / num * 100))


def normalize_images(dir_from, dir_to, omit=tuple(), max=float("inf")):
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

                image = Image.detect_face(image_from)
                if image is not None:
                    Image.save_image(image, image_to)

    # end :-)


if __name__ == "__main__":
    dir_from = "./apps_faces"
    dir_to = "./normalized_apps"

    if len(sys.argv) > 0:
        dir_from = sys.argv[0]
    if len(sys.argv) > 1:
        dir_to = sys.argv[1]

    normalize_images(dir_from, dir_to)
    test_all(dir_to)
