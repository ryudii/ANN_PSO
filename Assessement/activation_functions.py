import numpy as np


def null_activation_function():
    return 0


def sigmoid_activation_function(value):
    return 1/(1+np.exp(-value))


def hyperbolic_tangent_activation_function(value):
    return 1/(1+np.exp(-value))


def cosine_activation_function(value):
    return np.cos(value)


def gaussian_activation_function(value):
    return np.exp(-(value*value)/2)
