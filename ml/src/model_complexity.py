import torch


def analyze_model_complexity(model):

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    non_trainable_parameters = (
        total_parameters - trainable_parameters
    )

    total_layers = sum(
        1 for layer in model.modules()
        if len(list(layer.children())) == 0
    )

    model_size = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    model_size_mb = model_size / (1024 * 1024)

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "non_trainable_parameters": non_trainable_parameters,
        "total_layers": total_layers,
        "model_size_mb": round(model_size_mb, 4)
    }