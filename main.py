from processing import Eigenfaces, Image
import os
import sys


def test_person(eigen, file):
    probs = eigen.recognize(Image.read_image(file))
    name = max(probs, key=lambda x: x[1], default=["Failed"])[0]
    print("Detecting %s:" % file)
    print("\tMax:", name)
    print("\tProbabilities:", probs)
    print()

    return name == file.split("/")[0]


def test_detection_procedure(eigen, file):
    # Tests detection and save partial results
    image = Image.detect_face(file)
    image = image.flatten()

    if not os.path.exists("./detection_procedure/"):
        os.makedirs("./detection_procedure/")

    Image.save_image(eigen.average, "./detection_procedure/average.jpg")
    Image.save_image(image - eigen.average,
                     "./detection_procedure/subtracted.jpg")
    Image.save_image(eigen.reverse_image(eigen.calculate_weight(image))
                     - eigen.average,
                     "./detection_procedure/reduced-subtracted.jpg")
    Image.save_image(eigen.reverse_image(eigen.calculate_weight(image)),
                     "./detection_procedure/reduced.jpg")

    Image.show_image(image)


def save_eigenfaces(eigen):
    # Saves the eigenfaces to a folder
    if not os.path.exists("./eigenfaces/"):
        os.makedirs("./eigenfaces")

    for i in range(eigen.vectors.shape[1]):
        vector = eigen.vectors[:,i]
        vector *= 1 / max(vector)
        Image.save_image(vector + eigen.average, "./eigenvectors/" + str(i))


def playground(folder):
    eigen = Eigenfaces(folder)

    # EXAMPLES:
    # NOTE: parameters may change
    save_eigenfaces(eigen)
    test_person(eigen, folder + "/Andriy_Dmytruk/1.jpg")
    test_detection_procedure(eigen, folder + "/Andriy_Dmytruk/1.jpg")


if __name__ == "__main__":
    folder = "./normalized_apps"

    if len(sys.argv) > 1:
        com = sys.argv[1]

        if com == "server":
            dataset = "./apps_faces/"

            if len(sys.argv) > 2:
                folder = sys.argv[2]
            if len(sys.argv) > 3:
                dataset = sys.argv[3]

            from server import *
            initialize_server(folder, dataset)
            run_server()
        elif com == "detect":
            if len(sys.argv) > 2:
                name = sys.argv[2]
            else:
                raise Exception("Second parameter should be name")

            if len(sys.argv) > 3:
                folder = sys.argv[3]

            eigen = Eigenfaces(folder)
            test_person(eigen, sys.argv[2])
        elif com == "help":
            print("Available commands are: \n\thelp \n\tserver \n\ttest file_path [folder]")
        else:
            raise Exception("Invalid command. Available: help, serer, test")

    else:
        playground(folder)

