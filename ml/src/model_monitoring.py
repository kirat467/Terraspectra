def generate_model_monitoring(
    prediction_summary=None,
    confidence_analysis=None,
    drift_report=None,
    performance_report=None
):
    sections = {
        "prediction_summary": prediction_summary,
        "confidence_analysis": confidence_analysis,
        "drift_report": drift_report,
        "performance_report": performance_report
    }

    for name, data in sections.items():
        if data is not None and not isinstance(data, dict):
            raise ValueError(f"{name} must be a dictionary")

    recommendations = []

    if prediction_summary is None:
        recommendations.append("Prediction summary is unavailable.")

    if confidence_analysis is None:
        recommendations.append("Confidence analysis is unavailable.")

    if drift_report is None:
        recommendations.append("Model drift information is unavailable.")
        drift_status = "Unknown"
    else:
        drift_status = drift_report.get("status", "Unknown")

        if drift_status == "Drift Detected":
            recommendations.append(
                "Investigate prediction distribution changes."
            )
        elif drift_status == "Unknown":
            recommendations.append(
                "Review the model drift report."
            )
        else:
            recommendations.append(
                "Continue monitoring prediction distribution."
            )

    if performance_report is None:
        recommendations.append("Performance report is unavailable.")

    available_sections = sum(
        data is not None for data in sections.values()
    )

    return {
        "monitoring_status": (
            "Complete" if available_sections == 4 else "Partial"
        ),
        "available_sections": available_sections,
        "total_sections": 4,
        "drift_status": drift_status,
        "dashboard_data": sections,
        "recommendations": recommendations
    }