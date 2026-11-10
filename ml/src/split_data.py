
from sklearn.model_selection import train_test_split


def split_dataset(images, labels, test_size=0.2, val_size=0.2):

    if len(images) != len(labels):
        raise ValueError("Images and labels must have equal length")

    if len(images) < 3:
        raise ValueError("Dataset needs at least 3 samples")

    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("Invalid split sizes")

    X_train, X_temp, y_train, y_temp = train_test_split(
        images,
        labels,
        test_size=test_size + val_size,
        random_state=42
    )

    test_fraction = test_size / (test_size + val_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_fraction,
        random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
