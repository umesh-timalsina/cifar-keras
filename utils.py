from matplotlib import pyplot as plt
import numpy as np


def print_matrix_detail(matrix, name):
    """
    Prints the shape of the matrix

    Args:
        matrix: the numpy matrix
        name: the name of the matrix
    """
    print("Matrix {} is of shape {}".format(name, matrix.shape))

def _get_cifar_class(label):
    """Return Cifar classes from label"""
    cifar_classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]
    return cifar_classes[label]

def plot_image(matrix, labels):
    """
    Plot the image and the label
    """
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    fig = plt.figure(1)
    for i in range(len(matrix)):
        fig.add_subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(matrix[i], interpolation=None)
        plt.xlabel(_get_cifar_class(labels[i].squeeze()))
    plt.show()