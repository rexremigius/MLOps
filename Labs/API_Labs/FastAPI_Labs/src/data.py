from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load the California Housing dataset (1990 U.S. census, 20,640 block groups).

    The first call downloads the data (~400 KB) and caches it in
    ~/scikit_learn_data, so later runs work offline.

    Returns:
        X (numpy.ndarray): Feature matrix of shape (20640, 8).
        y (numpy.ndarray): Median house value per block group, in $100,000s.
        feature_names (list[str]): Column names of X, in order.
    """
    housing = fetch_california_housing()
    return housing.data, housing.target, list(housing.feature_names)


def split_data(X, y):
    """
    Split the data into training and testing sets (80/20).

    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)
