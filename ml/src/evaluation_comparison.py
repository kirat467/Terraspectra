import math
from numbers import Real


def compare_evaluations(model_a, model_b):

    required = ["accuracy", "precision", "recall", "f1_score"]

    validated_models = []

    for model in [model_a, model_b]:

        if not isinstance(model, dict):
            raise TypeError("Model evaluation must be a dictionary")

        validated_model = {}

        for key in required:

            if key not in model:
                raise ValueError(f"Missing metric: {key}")

            value = model[key]

            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{key} must be numeric")

            value = float(value)

            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")

            if not 0 <= value <= 1:
                raise ValueError(f"{key} must be between 0 and 1")

            validated_model[key] = value

        validated_models.append(validated_model)

    model_a, model_b = validated_models

    if model_a["f1_score"] > model_b["f1_score"]:
        winner = "Model A"

    elif model_b["f1_score"] > model_a["f1_score"]:
        winner = "Model B"

    elif model_a["accuracy"] > model_b["accuracy"]:
        winner = "Model A"

    elif model_b["accuracy"] > model_a["accuracy"]:
        winner = "Model B"

    else:
        winner = "Tie"

    return {
        "model_a_f1": model_a["f1_score"],
        "model_b_f1": model_b["f1_score"],
        "model_a_accuracy": model_a["accuracy"],
        "model_b_accuracy": model_b["accuracy"],
        "winner": winner
    }