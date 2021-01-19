"""Implement activation function on `n` inputs which is an arbitrary continuous piecewise-linear function on R^n with
non-differentiability only on the hyperplanes e_i = e_j (i.e., where two inputs are equal). Such a function
is determined by its value on the 2^n points of a hypercube. We restrict to functions which are 0 at the origin,
so that on each component the function is linear (and not only an affine function). The space of such functions
has a basis consisting of 2^n - 1 terms of the form x_i, min(x_i, x_j), min(x_i, x_j, x_k), ..., min(x_1, ..., x_n),
but we don't need to use this fact.
"""

import tensorflow as tf

def pack_bits(A):
    """Convert an array of booleans into ints by interpreting the booleans as bits in a binary representation.
    (There should be a faster way to do this without needing to do multiplications, but not sure if there is any
    support in TensorFlow yet.)"""
    n = A.shape[2]
    powers = 2 ** tf.range(n)
    return tf.reduce_sum(tf.expand_dims(tf.expand_dims(powers, 0), 1) * tf.cast(A, tf.int32), axis=2)


def high_order_act(A, params):
    A_mask = tf.expand_dims(A, 1) >= tf.expand_dims(A, 2)
    params_ind = pack_bits(A_mask)
    A_ind = tf.argsort(A, axis=1)
    A_sort = tf.gather(A, A_ind, batch_dims=1)
    A_diff = A_sort[:, 1:] - A_sort[:, :-1]
    coef = tf.concat([A_sort[:, 0:1], A_diff], axis=1)
    params_gather = tf.gather(params, tf.gather(params_ind, A_ind, batch_dims=1))
    out = tf.reduce_sum(coef * params_gather, axis=1)
    return out


# A = tf.random.normal([2, 4])
params = tf.random.normal([8])
A = tf.constant([[0.5, 0.5, 0.5]])
print(high_order_act(A, params), params)

# def high_order_act(A, params):
#     A_ind = tf.argsort(A, axis=0)
