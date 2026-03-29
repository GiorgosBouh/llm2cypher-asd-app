from __future__ import annotations

from typing import Iterable


def confidence_summary(probability: float, high: float, medium: float) -> str:
    """Convert probability into a simple confidence label."""
    certainty = max(probability, 1.0 - probability)
    if certainty >= high:
        return "High confidence"
    if certainty >= medium:
        return "Moderate confidence"
    return "Low confidence"


def prediction_plain_language(top_features: Iterable[str], predicted_label: str, probability_yes: float) -> str:
    """Build a short human-readable explanation of the prediction."""
    predicted_probability = probability_yes if predicted_label == "Yes" else 1.0 - probability_yes
    features = [feature.replace("_", " ") for feature in top_features if feature]
    if not features:
        return (
            f"The app classified this case as {predicted_label}. "
            f"The probability of the predicted label was {predicted_probability:.1%}, "
            f"while the model-estimated probability for ASD traits = Yes was {probability_yes:.1%}. "
            "A clearer feature-by-feature explanation was not available."
        )

    preview = ", ".join(features[:4])
    return (
        f"This result was affected most by {preview}. "
        f"Overall, the app classified this case as {predicted_label}. "
        f"The probability of the predicted label was {predicted_probability:.1%}, "
        f"while the model-estimated probability for ASD traits = Yes was {probability_yes:.1%}."
    )


def anomaly_plain_language(score: float, is_anomalous: bool) -> str:
    """Explain anomaly score to a non-technical user."""
    status = "unusual" if is_anomalous else "within the learned range"
    return (
        f"The profile appears {status} relative to the training data. "
        f"Supportive anomaly score: {score:.3f}. "
        "Anomaly detection is supportive and does not determine the ASD prediction."
    )
