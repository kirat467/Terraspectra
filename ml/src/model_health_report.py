def generate_model_health_report(health_data):
    required = [
        "health_status",
        "health_score",
        "performance_metrics",
        "average_confidence",
        "drift_detected",
        "recommendations"
    ]

    for key in required:
        if key not in health_data:
            raise ValueError(f"Missing required field: {key}")

    metrics = health_data["performance_metrics"]

    report = []
    report.append("TERRASPECTRA - MODEL HEALTH REPORT")
    report.append("=" * 40)

    report.append(f"Health Status: {health_data['health_status']}")
    report.append(f"Health Score: {health_data['health_score']}%")

    report.append("\nPERFORMANCE METRICS")
    report.append("-" * 25)

    for metric, value in metrics.items():
        report.append(f"{metric.replace('_', ' ').title()}: {value}")

    report.append("\nAVERAGE CONFIDENCE")
    report.append(str(health_data["average_confidence"]))

    report.append("\nMODEL DRIFT")
    report.append(
        "Detected" if health_data["drift_detected"] else "Not Detected"
    )

    report.append("\nRECOMMENDATIONS")
    report.append("-" * 25)

    if health_data["recommendations"]:
        for index, recommendation in enumerate(
            health_data["recommendations"], 1
        ):
            report.append(f"{index}. {recommendation}")
    else:
        report.append("No recommendations. Model health is satisfactory.")

    return "\n".join(report)