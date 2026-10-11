import tensorflow as tf
import numpy as np

from tensorflow.keras.applications.efficientnet import preprocess_input

#Image loading
def load_single_image(
    image_path,
    image_size=(224, 224),
    model_name="cnn"
):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        image_size
    )

    image = tf.cast(image, tf.float32)

    if model_name.lower() == "efficientnetb0":
        image = preprocess_input(image)
    else:
        image = image / 255.0

    image = tf.expand_dims(image, axis=0)

    return image


#Predictions
def predict_image(
    model_name,
    image_path,
    image_size=(224, 224)
):

    MODEL_PATH = f"models/{model_name}.keras"

    print(f"\nLoading model: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    image = load_single_image(
        image_path=image_path,
        image_size=image_size,
        model_name=model_name
    )

    probability = model.predict(
        image,
        verbose=0
    )[0][0]

    if probability >= 0.5:
        predicted_class = "Spiral (S)"
    else:
        predicted_class = "Smooth (E)"

    print("Prediction:")
    print("\n")
    print(f"Image      : {image_path}")
    print(f"Model      : {model_name}")
    print(f"Prediction : {predicted_class}")
    print(f"Probability: {probability:.4f}")




if __name__ == "__main__":

    #Model path and image path (change for a single prediction)
    MODEL_NAME = "efficientnetb0"
    IMAGE_PATH = "D:/Downloads_3/GZ_2/imgs/images/137481.jpg"

    predict_image(
        model_name=MODEL_NAME,
        image_path=IMAGE_PATH
    )
