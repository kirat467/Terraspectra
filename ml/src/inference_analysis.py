import time
import torch


def analyze_inference_time(model, dataloader, device=None, warmup=2):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("Warmup must be a non-negative integer")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    model = model.to(device)
    model.eval()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    total_samples = 0
    total_batches = 0
    total_time = 0.0

    with torch.inference_mode():

        # Warmup runs are excluded from timing
        iterator = iter(dataloader)

        for _ in range(warmup):
            try:
                images, _ = next(iterator)
            except StopIteration:
                break

            images = images.to(device)
            model(images)

        synchronize()

        # Measure actual inference
        for images, _ in dataloader:

            images = images.to(device)

            synchronize()
            start = time.perf_counter()

            model(images)

            synchronize()
            elapsed = time.perf_counter() - start

            total_time += elapsed
            total_samples += images.shape[0]
            total_batches += 1

    if total_samples == 0:
        raise ValueError("Inference dataset is empty")

    return {
        "total_samples": total_samples,
        "total_batches": total_batches,
        "total_inference_time": round(total_time, 6),
        "average_batch_time": round(
            total_time / total_batches, 6
        ),
        "average_sample_time": round(
            total_time / total_samples, 6
        ),
        "samples_per_second": round(
            total_samples / total_time, 2
        ) if total_time > 0 else None,
        "device": str(device)
    }