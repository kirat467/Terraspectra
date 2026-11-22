def analyze_prediction_stability(predictions):
    if not isinstance(predictions, list):
        raise TypeError("Predictions must be a list")

    if not predictions:
        raise ValueError("Predictions cannot be empty")

    results = []

    for index, sample in enumerate(predictions):
        if not isinstance(sample, (list, tuple)):
            raise TypeError("Each sample must contain a list of predictions")

        if not sample:
            raise ValueError("Each sample must have at least one prediction")

        if any(value is None for value in sample):
            raise ValueError("Predictions cannot contain None")

        counts = {}

        for value in sample:
            counts[value] = counts.get(value, 0) + 1

        majority_count = max(counts.values())
        stability = majority_count / len(sample)

        results.append({
            "sample_index": index,
            "unique_predictions": len(counts),
            "stability_score": round(stability, 4),
            "stable": len(counts) == 1
        })

    average_stability = sum(
        item["stability_score"] for item in results
    ) / len(results)

    return {
        "sample_results": results,
        "average_stability": round(average_stability, 4),
        "total_samples": len(results)
    }