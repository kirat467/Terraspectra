
import numpy as np
from split_data import split_dataset

images = np.random.rand(100, 16, 16, 200)
labels = np.random.randint(0, 3, 100)

result = split_dataset(images, labels)

X_train, X_val, X_test, y_train, y_val, y_test = result

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

print("Dataset splitting test passed!")
