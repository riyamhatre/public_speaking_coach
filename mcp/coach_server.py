from mcp.server import Server
from mcp.server.stdio import stdio_server
from ml.delivery_model import predict_delivery

server = Server("public-speaking-coach")


@server.tool()
def predict_delivery_score(
    pace_wpm: float,
    filler_count: int,
    avg_pause: float,
    duration: float
) -> dict:
    """
    Predict speaker delivery confidence (0–1).
    """
    score = predict_delivery([
        pace_wpm,
        filler_count,
        avg_pause,
        duration
    ])

    return {
        "delivery_confidence": round(score, 2)
    }


if __name__ == "__main__":
    stdio_server(server)
