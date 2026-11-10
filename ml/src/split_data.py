
import numpy as np
from split_data import split_dataset

images = np.random.rand(100, 16, 16, 200)
labels = np.random.randint(0, 3, 100)

result = split_dataset(images, labels)

X_train, X_val, X_test, y_train, y_val, y_test = result

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

assert len(X_train) == 60
assert len(X_val) == 20
assert len(X_test) == 20

print("All dataset split tests passed!")
