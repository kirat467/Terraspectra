
import numpy as np
from analyze_dataset import analyze_dataset

images = np.random.rand(100, 16, 16, 200)

labels = (
    ["Healthy"] * 40
    + ["Leaf Rust"] * 35
    + ["Blight"] * 25
)

report = analyze_dataset(images, labels)

assert len(report) == 3
assert sum(report.values()) == 100

print("\nDataset analysis test passed!")
