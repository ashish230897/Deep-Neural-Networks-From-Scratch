import numpy as np
import math
from PIL import Image
import os

def random_mini_batches(x, y, mini_batch_size=64, seed=0):
    np.random.seed(seed)

    m = x.shape[1]  # Training examples length
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_x = x[:, permutation]
    shuffled_y = y[:, permutation].reshape((1, m))

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_x = shuffled_x[:, k * (mini_batch_size): (k + 1) * (mini_batch_size)]
        mini_batch_y = shuffled_y[:, k * (mini_batch_size): (k + 1) * (mini_batch_size)]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_x = shuffled_x[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_y = shuffled_y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
    return mini_batches


def shuffle(a, b):
    permutation = list(np.random.permutation(a.shape[1]))
    a = a[:, permutation]
    b = b[:, permutation].reshape((1, a.shape[1]))
    return a, b


def initialize_parameters_he(layer_dims, load_pretrained):
    parameters = {}

    # number of layers in the network
    num_layers = len(layer_dims)

    if (load_pretrained):  # If loading pretrained parameters
        for l in range(1, num_layers):
            parameters['W' + str(l)] = np.array(np.load("parametersW" + str(l) + ".npy"))
            parameters['b' + str(l)] = np.array(np.load("parametersb" + str(l) + ".npy"))

            assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters

    # He initialization works well for networks with ReLU activations.
    # As the length of layer dims goes from 0 to L-1 , we go from 1 to L-1 as 0th layer is i/p layer
    # The initialization is done this way to avoid exploding and diminishing of output of each layer
    for l in range(1, num_layers):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def initialize_velocity_momentum(parameters):
    v = {}
    num_layers = len(parameters) // 2  # Number of layers

    for i in range(1, num_layers + 1):
        v["dw" + str(i)] = np.zeros((parameters["W" + str(i)].shape[0], parameters["W" + str(i)].shape[1]))
        v["db" + str(i)] = np.zeros((parameters["b" + str(i)].shape[0], parameters["b" + str(i)].shape[1]))

        assert (v["dw" + str(i)].shape == parameters["W" + str(i)].shape)
        assert (v["db" + str(i)].shape == parameters["b" + str(i)].shape)

    return v


def update_parameters_with_momentum(parameters, grads, learning_rate, v, beta):
    """
    Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples,
    the direction of the update has some variance, and so the path taken by mini-batch gradient descent will
    "oscillate" toward convergence. Using momentum can reduce these oscillations.

    Momentum takes into account the past gradients to smooth out the update. We will store the 'direction'
    of the previous gradients in the variable v . Formally, this will be the exponentially weighted average
    of the gradient on previous steps. You can also think of v as the "velocity" of a ball rolling downhill,
    building up speed (and momentum) according to the direction of the gradient/slope of the hill.
    """
    num_layers = len(parameters) // 2

    # Update rule for each parameter.
    for l in range(1, num_layers + 1):
        v["dw" + str(l)] = beta * v["dw" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
        parameters["W" + str(l)] -= (learning_rate * v["dw" + str(l)])
        parameters["b" + str(l)] -= (learning_rate * v["db" + str(l)])

    return parameters, v


def load_data(path):
    image_names = os.listdir(path)  # Put all the training images at this path

    train_images = []
    train_labels = []
    for img_ in image_names:
        img1 = Image.open(path + img_)
        img2 = np.array(img1.resize((64, 64), Image.ANTIALIAS))  # Resizing each image to (64, 64, 3)
        train_images.append(img2)
        if "cat" in img_:
            train_labels.append(0)  # each training image has 'cat' or 'dog' in it's name
        else:
            train_labels.append(1)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    train_labels = np.expand_dims(train_labels, axis=-1)

    return train_images, train_labels


def preprocess(train_x_orig, train_y_orig):
    # Reshaping the images to vectors of following form (num_features, num_training_examples)
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    train_y = train_y_orig.reshape(train_y_orig.shape[0], -1).T

    # Normalizing values
    train_x = train_x_flatten / 255

    train_x, train_y = shuffle(train_x, train_y)  # shuffling the training images and labels

    return train_x, train_y