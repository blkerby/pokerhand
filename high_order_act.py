"""Implement activation function on `n` inputs which is an arbitrary continuous piecewise-linear function on R^n with
non-differentiability only on the hyperplanes e_i = e_j (i.e., where two inputs are equal). We restrict to functions
which are 0 at the origin, so that on each component the function is linear (and not only an affine function). The
space of such functions has a basis consisting of 2^n - 1 terms of the form x_i, min(x_i, x_j), min(x_i, x_j, x_k),
..., min(x_1, ..., x_n), but we don't need to use this fact. Instead, we use the fact that such a function is
determined by its value on the 2^n - 1 vertices of the hypercube {0, 1}^n excluding the origin; the hyperplanes
e_i = e_j partition the hypercube [0, 1]^n into a simplicial complex (with n! simplices, one for each possible ordering
of the components x_i)) with the points {0, 1}^n being the vertices; any function on the vertex set then extends
uniquely to a continuous, piecewise-linear function on the simplicial complex, linear on each simplex.

The motivation is that this allows us to implement a higher-order interaction between `n` variables in a single layer.
For instance, for inputs in the set {0, 1} an arbitrary Boolean function in the `n` variables can be represented. Also,
the number of parameters grows exponentially as 2^n - 1, yet the computation required for an inference or training pass
only grows linearly with `n`; this is because only `n` of the parameters need to be accessed for any given example.
"""

import tensorflow as tf

def pack_bits(A):
    """Convert an array of booleans into ints by interpreting the booleans as bits in a binary representation.
    (There should be a faster way to do this without needing to do multiplications, but not sure if there is any
    support in TensorFlow yet.)"""
    n = A.shape[3]
    powers = 2 ** tf.range(n)
    return tf.reduce_sum(tf.expand_dims(tf.expand_dims(tf.expand_dims(powers, 0), 1), 2) * tf.cast(A, tf.int32), axis=3)


def high_order_act(A, params):
    A_mask = tf.expand_dims(A, 2) >= tf.expand_dims(A, 3)
    params_ind = pack_bits(A_mask)
    A_ind = tf.argsort(A, axis=2)
    A_sort = tf.gather(A, A_ind, batch_dims=2)
    A_diff = A_sort[:, :, 1:] - A_sort[:, :, :-1]
    coef = tf.concat([A_sort[:, :, 0:1], A_diff], axis=2)
    params_A_ind = tf.gather(params_ind, A_ind, batch_dims=2)
    params_gather = tf.gather(params, tf.transpose(params_A_ind, perm=[1, 0, 2]), batch_dims=1)
    out = tf.einsum('jikl,ijk->ijl', params_gather, coef)
    return out



def fast_high_order_act(A, params):
    A_ind = tf.argsort(A, axis=2)
    A_sort = tf.gather(A, A_ind, batch_dims=2)
    A_diff = A_sort[:, :, 1:] - A_sort[:, :, :-1]
    coef = tf.concat([A_sort[:, :, 0:1], A_diff], axis=2)
    params_A_ind = tf.cumsum(tf.bitwise.left_shift(1, A_ind), reverse=True, axis=2)
    params_gather = tf.gather(params, tf.transpose(params_A_ind, perm=[1, 0, 2]), batch_dims=1)
    out = tf.einsum('jikl,ijk->ijl', params_gather, coef)
    return out


class HighOrderActivation(tf.keras.layers.Layer):
    def __init__(self, arity, out_dim):
        super().__init__()
        self.arity = arity
        self.out_dim = out_dim

    def build(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[2] == self.arity
        self.input_dim = input_shape[1]
        self.params = tf.Variable(tf.random.normal([self.input_dim, 2 ** self.arity, self.out_dim]))

    def call(self, X):
        assert len(X.shape) == 3
        assert X.shape[1] == self.input_dim
        assert X.shape[2] == self.arity
        # return high_order_act(X, self.params)
        return fast_high_order_act(X, self.params)



class BatchHighOrderActivation(tf.keras.layers.Layer):
    def __init__(self, arity, out_dim):
        super().__init__()
        self.arity = arity
        self.out_dim = out_dim

    def build(self, input_shape):
        assert input_shape[-1] == self.arity
        self.input_dim = input_shape[-2]
        self.params = tf.Variable(tf.random.normal([self.input_dim, 2 ** self.arity, self.out_dim]))

    def call(self, X):
        X1 = tf.reshape(X, [-1, self.input_dim, self.arity])
        # Y1 = high_order_act(X1, self.params)
        Y1 = fast_high_order_act(X1, self.params)
        Y_shape = list(X.shape)
        Y_shape[0] = -1
        Y_shape[-1] = self.out_dim
        Y = tf.reshape(Y1, tf.constant(Y_shape))
        return Y



# m = 2
# n = 4
# k = 4
# params = tf.random.normal([k, 2 ** n, 4])
# A = tf.random.normal([m, k, n])
# out = high_order_act(A, params)
# out2 = fast_high_order_act(A, params)
# print(out - out2)