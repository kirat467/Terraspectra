
from sklearn.model_selection import train_test_split


def split_dataset(images, labels, test_size=0.2, val_size=0.2):
    """
    Split data into training, validation and testing sets.
    """

    if len(images) != len(labels):
        raise ValueError("Images and labels must have equal length")

    if len(images) < 3:
        raise ValueError("Dataset needs at least 3 samples")

    # First separate training data and temporary data
    X_train, X_temp, y_train, y_temp = train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=42
    )

    # Split temporary data into validation and testing
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
