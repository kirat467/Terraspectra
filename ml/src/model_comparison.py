import math


METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]


def compare_models(cnn_results, vit_results):

    models = {
        "3D-CNN": cnn_results,
        "ViT": vit_results
    }

    # Validate evaluation results
    for model_name, results in models.items():

        if not isinstance(results, dict):
            raise ValueError(f"Invalid results for {model_name}")

        for metric in METRICS:

            value = results.get(metric)

            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    f"Invalid {metric} for {model_name}"
                )

    comparison = {}

    # Compare all metrics
    for metric in METRICS:

        cnn_value = cnn_results[metric]
        vit_value = vit_results[metric]

        if cnn_value > vit_value:
            winner = "3D-CNN"
        elif vit_value > cnn_value:
            winner = "ViT"
        else:
            winner = "Tie"

        comparison[metric] = {
            "3D-CNN": cnn_value,
            "ViT": vit_value,
            "difference": round(cnn_value - vit_value, 4),
            "winner": winner
        }

    # Select best model using F1-score
    if cnn_results["f1_score"] > vit_results["f1_score"]:
        best_model = "3D-CNN"

    elif vit_results["f1_score"] > cnn_results["f1_score"]:
        best_model = "ViT"

    else:
        best_model = "Tie"

    return {
        "model_comparison": comparison,
        "best_model": best_model,
        "selection_metric": "f1_score"
    }