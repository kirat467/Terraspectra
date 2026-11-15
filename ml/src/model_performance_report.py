from model_complexity import analyze_model_complexity


def generate_model_performance_report(
    model,
    evaluation_results,
    inference_results,
    model_name="TerraSpectra Model"
):

    required_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]

    for metric in required_metrics:
        if metric not in evaluation_results:
            raise ValueError(f"Missing evaluation metric: {metric}")

    if not model_name or not isinstance(model_name, str):
        raise ValueError("Model name must be a non-empty string")

    if not isinstance(inference_results, dict):
        raise ValueError("Inference results must be a dictionary")

    complexity = analyze_model_complexity(model)

    return {
        "model_name": model_name,

        "performance": {
            "accuracy": evaluation_results["accuracy"],
            "precision": evaluation_results["precision"],
            "recall": evaluation_results["recall"],
            "f1_score": evaluation_results["f1_score"],
            "mean_confidence": evaluation_results.get(
                "mean_confidence"
            )
        },

        "inference": inference_results,

        "complexity": complexity,

        "summary": {
            "total_parameters": complexity["total_parameters"],
            "model_size_mb": complexity["model_size_mb"],
            "inference_speed": inference_results.get(
                "samples_per_second"
            ),
            "evaluation_samples": inference_results.get(
                "total_samples"
            )
        }
    }