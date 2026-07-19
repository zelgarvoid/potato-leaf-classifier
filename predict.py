import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


mobilenet_model = tf.keras.models.load_model("models/mobilenet.keras")
efficientnet_model = tf.keras.models.load_model("models/efficientnet.keras")


class_names = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]


disease_info = {
    "Early Blight": {
        "description": "Disebabkan oleh jamur Alternaria solani.",
        "solution": "Gunakan fungisida dan buang daun yang terinfeksi."
    },
    "Late Blight": {
        "description": "Disebabkan oleh Phytophthora infestans.",
        "solution": "Gunakan fungisida berbahan aktif dan hindari kelembapan tinggi."
    },
    "Healthy": {
        "description": "Daun kentang dalam kondisi sehat.",
        "solution": "Lanjutkan perawatan tanaman seperti biasa."
    }
}


def predict_image(image_path, model_name):

    if model_name == "efficientnet":
        model = efficientnet_model
    else:
        model = mobilenet_model

    img = image.load_img(image_path, target_size=(224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    confidence = float(np.max(predictions) * 100)

    predicted_class = class_names[np.argmax(predictions)]

    return {
        "class": predicted_class,
        "confidence": round(confidence, 2),
        "description": disease_info[predicted_class]["description"],
        "solution": disease_info[predicted_class]["solution"]
    }