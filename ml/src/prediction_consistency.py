def analyze_prediction_consistency(predictions):
    if not isinstance(predictions, list):
        raise TypeError("Predictions must be a list")

    if not predictions:
        raise ValueError("Predictions cannot be empty")

    results = []

    for index, sample_predictions in enumerate(predictions):
        if not isinstance(sample_predictions, (list, tuple)):
            raise TypeError("Each sample must contain a list of predictions")

        if not sample_predictions:
            raise ValueError("Each sample must have at least one prediction")

        if any(prediction is None for prediction in sample_predictions):
            raise ValueError("Predictions cannot contain None")

        counts = {}

        for prediction in sample_predictions:
            counts[prediction] = counts.get(prediction, 0) + 1

        most_common = max(counts, key=counts.get)
        consistency = counts[most_common] / len(sample_predictions)

        results.append({
            "sample_index": index,
            "most_common_prediction": most_common,
            "consistency_score": round(consistency, 4),
            "consistent": consistency == 1.0
        })

    average_consistency = sum(
        item["consistency_score"] for item in results
    ) / len(results)

    return {
        "sample_results": results,
        "average_consistency": round(average_consistency, 4),
        "total_samples": len(results)
    }