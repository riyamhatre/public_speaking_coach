import tensorflow as tf
import numpy as np
from pathlib import Path

MODEL_PATH = Path("ml/delivery_model.keras")


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def load_or_create_model():
    if MODEL_PATH.exists():
        return tf.keras.models.load_model(MODEL_PATH)

    model = build_model()
    model.save(MODEL_PATH)
    return model


def predict_delivery(features):
    """
    features = [pace_wpm, filler_count, avg_pause, duration]
    """
    model = load_or_create_model()
    prediction = model.predict(np.array([features]), verbose=0)
    return float(prediction[0][0])
