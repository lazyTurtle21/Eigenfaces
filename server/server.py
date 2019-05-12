import base64
from flask import Flask, Blueprint, \
    jsonify, request, render_template, redirect
from time import time as timestamp
from processing import Image, Eigenfaces

app = Flask(__name__)
app.url_map.strict_slashes = False

blueprint = Blueprint("eigenfaces", __name__)
eigen = Eigenfaces("../normalized_apps")


@blueprint.route("/")
def landing():
    return render_template("index.html")


@blueprint.route("/detect", methods=["POST"])
def recognize_image():
    image = request.data.split(b",")[1]

    # omit "data:image/png;base64,"
    id = int(timestamp() * 1e6)
    filename = "./images/{}.{}".format(id, "jpg")

    # also String IO can be used to load image
    with open(filename, "wb") as file:
        file.write(base64.decodebytes(image + b'=' * (-len(image) % 4)))

    probs = detect_face(filename)
    return jsonify({"probabilities": probs}), 200


def detect_face(filename):
    """ Function takes file path and returns name """
    image = Image.find_face(filename)
    if image is None:
        return []

    Image.show_image(image)
    probs = eigen.recognize(image.flatten(), "norm")

    return probs


app.route("/")(lambda: redirect("/eigenfaces"))
app.register_blueprint(blueprint, url_prefix="/eigenfaces")


if __name__ == '__main__':
    app.run(debug=True)

