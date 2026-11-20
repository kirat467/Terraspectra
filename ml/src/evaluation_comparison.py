def compare_evaluations(model_a, model_b):
    required = ["accuracy", "precision", "recall", "f1_score"]

    for model in [model_a, model_b]:
        if not isinstance(model, dict):
            raise TypeError("Model evaluation must be a dictionary")

        for key in required:
            if key not in model:
                raise ValueError(f"Missing metric: {key}")

            value = model[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{key} must be numeric")

            if not 0 <= value <= 1:
                raise ValueError(f"{key} must be between 0 and 1")

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