import numpy as np


def relu_backward(da, activation_cache):
	# Since dz = da * g'(z)
	z = activation_cache["Z"]

	dz = np.multiply(da, np.int64(z > 0))

	assert (z.shape == dz.shape)
	return dz


def sigmoid_backward(da, activation_cache):
	# Since dz = da * g'(z)
	z = activation_cache["Z"]
	a = 1 / (1 + np.exp(-1 * z))

	dz = np.multiply(da, a * (1 - a))

	assert (z.shape == dz.shape)
	return dz


def linear_activation_backward(da, cache, activation, lambd):
	# cache is of a particular layer, this function is run for every layer to calculate dA_prev for preceding layer
	linear_cache, activation_cache = cache

	# To calculate dz we need da
	# dz of current layer is necessary to calculate every other gradient
	if activation == "relu":
		dz = relu_backward(da, activation_cache)
		da_prev, dw, db = linear_backward(dz, linear_cache, lambd)

	if activation == "sigmoid":
		dz = sigmoid_backward(da, activation_cache)
		da_prev, dw, db = linear_backward(dz, linear_cache, lambd)

	# da_prev is gradient of previous layer's output that is a_prev
	return da_prev, dw, db


def linear_backward(dz, cache, lambd):
    # We need dz of any layer before calculating other gradients of that layer
    # dz is caculated from dA , which is then used to calculate dA_prev

    a_prev, w, b = cache
    m = a_prev.shape[1]

    dw = (1/m)*(np.dot(dz,a_prev.T)) + (lambd/m)*w
    db = (1/m)*(np.sum(dz, axis = 1, keepdims = True))
    da_prev = np.dot(w.T,dz)

    assert (da_prev.shape == a_prev.shape)
    assert (dw.shape == w.shape)
    assert (db.shape == b.shape)

    return da_prev, dw, db