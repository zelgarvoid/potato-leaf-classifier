import os

from flask import Flask, render_template, request

from werkzeug.utils import secure_filename

from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    image_path = None

    if request.method == "POST":

        file = request.files["image"]

        model = request.form["model"]

        filename = secure_filename(file.filename)

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        file.save(filepath)

        result = predict_image(filepath, model)

        image_path = filepath.replace("\\", "/")

    return render_template(
        "index.html",
        result=result,
        image=image_path
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)