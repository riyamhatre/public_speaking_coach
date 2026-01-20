import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path("ml/delivery_data.csv")

def generate_sample_data(n_samples=1000):
    """
    Generates synthetic public speaking delivery data and saves as CSV.
    """
    # Generate features
    pace_wpm = np.random.normal(loc=140, scale=20, size=n_samples)
    filler_count = np.random.poisson(lam=5, size=n_samples)
    avg_pause = np.random.normal(loc=0.7, scale=0.3, size=n_samples)
    duration = np.random.normal(loc=120, scale=30, size=n_samples)

    # Clip to realistic limits
    pace_wpm = np.clip(pace_wpm, 80, 200)
    filler_count = np.clip(filler_count, 0, 20)
    avg_pause = np.clip(avg_pause, 0.2, 2.0)
    duration = np.clip(duration, 30, 300)

    # Generate delivery score
    score = (
        0.4 * (1 - np.abs(pace_wpm - 140)/60) +
        0.3 * (1 - filler_count/20) +
        0.2 * (1 - avg_pause/2) +
        0.1 * (1 - np.abs(duration - 120)/180)
    )
    score = np.clip(score, 0, 1)

    # Create DataFrame
    data = pd.DataFrame({
        "pace_wpm": pace_wpm,
        "filler_count": filler_count,
        "avg_pause": avg_pause,
        "duration": duration,
        "score": score
    })

    # Ensure folder exists
    DATA_PATH.parent.mkdir(exist_ok=True)
    data.to_csv(DATA_PATH, index=False)
    print(f"Sample training data saved to {DATA_PATH}")

if __name__ == "__main__":
    generate_sample_data()