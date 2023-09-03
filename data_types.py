import numpy as np


def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

def generate_vertical_data(samples_per_class, num_classes):
    """
    Generates vertical data with specified samples and classes.

    Args:
        samples_per_class (int): Number of samples per class.
        num_classes (int): Number of classes.

    Returns:
        data (ndarray): Generated dataset.
        labels (ndarray): Corresponding labels for the dataset.
    """
    np.random.seed(0)
    num_features = 2  # Number of features
    noise_level = 0.1

    data = []
    labels = []

    for class_label in range(num_classes):
        class_center = np.random.randn(num_features)
        class_data = class_center + np.random.randn(samples_per_class, num_features) * noise_level
        data.append(class_data)
        labels.extend([class_label] * samples_per_class)

    data = np.vstack(data)
    labels = np.array(labels)

    return data, labels