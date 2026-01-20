import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("ml/delivery_model.keras")
DATA_PATH = Path("ml/delivery_data.csv")


def build_model():
    """
    Creates a simple feed-forward neural network for delivery scoring.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",   # regression problem
        metrics=["mae"]
    )
    return model


def load_or_create_model(train_if_missing=True):
    if MODEL_PATH.exists():
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print("⚠️ Failed to load model, retraining:", e)
            MODEL_PATH.unlink(missing_ok=True)

    model = build_model()

    if train_if_missing:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Training data CSV not found at {DATA_PATH}"
            )

        data = pd.read_csv(DATA_PATH)
        X_train = data[["pace_wpm", "filler_count", "avg_pause", "duration"]].values
        y_train = data["score"].values

        model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_PATH)

    return model



def predict_delivery(features):
    """
    Predicts delivery score for a given speech sample.
    features = [pace_wpm, filler_count, avg_pause, duration]
    returns: float between 0 and 1
    """
    model = load_or_create_model()
    prediction = model.predict(np.array([features]), verbose=0)
    return float(prediction[0][0])


# Optional: Test
if __name__ == "__main__":
    sample_features = [130, 3, 0.7, 90]
    score = predict_delivery(sample_features)
    print(f"Predicted delivery score: {score:.2f}")


