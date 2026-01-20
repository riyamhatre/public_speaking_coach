import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = Path("ml/delivery_model.keras")
DATA_PATH = Path("ml/delivery_data.csv")


def build_model():
    """
    Creates a feed-forward neural network for delivery scoring.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")  # output between 0 and 1
    ])
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",   # regression problem
        metrics=["mae"]
    )
    return model


def load_or_create_model(train_if_missing=True):
    """
    Loads the model from disk, or creates and trains it if not present.
    """
    if MODEL_PATH.exists():
        return tf.keras.models.load_model(MODEL_PATH)

    model = build_model()

    # Train on CSV if available
    if train_if_missing:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Training data CSV not found at {DATA_PATH}. Run generate_training_data.py first.")

        # Read CSV
        data = pd.read_csv(DATA_PATH)
        X = data[["pace_wpm", "filler_count", "avg_pause", "duration"]].values
        y = data["score"].values

        # Normalize features to 0-1 range
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Save scaler for later use
        scaler_path = Path("ml/scaler.npy")
        np.save(scaler_path, scaler.scale_)
        np.save(Path("ml/scaler_min.npy"), scaler.min_)

        # Train model
        model.fit(X_scaled, y, epochs=50, batch_size=32, verbose=1)

        MODEL_PATH.parent.mkdir(exist_ok=True)
        model.save(MODEL_PATH)

    return model


def predict_delivery(features):
    """
    Predicts delivery score for a given speech sample.
    features = [pace_wpm, filler_count, avg_pause, duration]
    returns: float between 0 and 1
    """
    model = load_or_create_model()

    # Load scaler to normalize input
    scale = np.load(Path("ml/scaler.npy"))
    min_ = np.load(Path("ml/scaler_min.npy"))
    features_scaled = (np.array(features) - min_) / scale

    prediction = model.predict(np.array([features_scaled]), verbose=0)
    return float(prediction[0][0])


# Optional test
if __name__ == "__main__":
    sample_features = [130, 3, 0.7, 90]
    score = predict_delivery(sample_features)
    print(f"Predicted delivery score: {score:.2f}")
