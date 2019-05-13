import base64
import re
import os
from flask import Flask, Blueprint, \
    jsonify, request, render_template, redirect
from time import time as timestamp
from processing import Image, Eigenfaces

UPLOAD_FOLDER = "./images"
DATASET_FOLDER = "../Dataset"

app = Flask(__name__)
app.url_map.strict_slashes = False

blueprint = Blueprint("eigenfaces", __name__)
eigen = Eigenfaces("../normalized_apps")


@blueprint.route("/")
def landing():
    return render_template("index.html")


@blueprint.route("/save")
def save_landing():
    return render_template("save.html")


def create_filename(format="jpg", folder=None):
    folder = folder or UPLOAD_FOLDER
    if not os.path.isdir(folder):
        os.makedirs(folder)

    id = int(timestamp() * 1e6)
    filename = "{}/{}.{}".format(folder, id, format)

    return filename


def save_body_image(folder=None):
    # request body example: "data:image/png;base64,"
    data = request.data

    matcher = re.search(b"^data:image/([^;]*);base64,(.*)$", data)

    if not matcher:
        return jsonify({"probabilities": [], "error": "Invalid format"}), 422

    format = matcher.group(1)
    image = matcher.group(2)

    filename = create_filename(format.decode('ascii'), folder)
    # also String IO can be used to load image
    with open(filename, "wb") as file:
        file.write(base64.decodebytes(image + b'=' * (-len(image) % 4)))

    return filename


@blueprint.route("/detect/data", methods=["POST"])
def recognize_image():
    filename = save_body_image()

    probs = detect_face(filename)
    return jsonify({"probabilities": probs}), 200


@blueprint.route("/save/<name>", methods=["POST"])
def save_image(name):
    folder = DATASET_FOLDER + "/" + name
    save_body_image(folder)
    return "", 200


def detect_face(filename):
    """ Function takes file path and returns name """
    image = Image.find_face(filename)
    if image is None:
        return []

    # Image.show_image(image)
    probs = eigen.recognize(image.flatten(), "norm")

    return probs


app.route("/")(lambda: redirect("/eigenfaces"))
app.register_blueprint(blueprint, url_prefix="/eigenfaces")


if __name__ == '__main__':
    app.run(debug=True, port=4000)

