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


@tf.function
def fast_high_order_act(A, params):
    A_ind = tf.argsort(A, axis=2)
    A_sort = tf.gather(A, A_ind, batch_dims=2)
    A_diff = A_sort[:, :, 1:] - A_sort[:, :, :-1]
    coef = tf.concat([A_sort[:, :, 0:1], A_diff], axis=2)
    params_A_ind = tf.cumsum(tf.bitwise.left_shift(1, A_ind), reverse=True, axis=2)
    params_gather = tf.gather(params, tf.transpose(params_A_ind, perm=[1, 0, 2]), batch_dims=1)
    out = tf.einsum('jikl,ijk->ijl', params_gather, coef)
    return out


@tf.function
def sparsemm_high_order_act(A, params):
    A_ind = tf.argsort(A, axis=2)
    A_sort = tf.gather(A, A_ind, batch_dims=2)
    A_diff = A_sort[:, :, 1:] - A_sort[:, :, :-1]
    coef = tf.concat([A_sort[:, :, 0:1], A_diff], axis=2)
    ind = tf.cumsum(tf.bitwise.left_shift(1, A_ind), reverse=True, axis=2)
    ind2 = tf.expand_dims(tf.expand_dims(tf.range(tf.shape(A)[1]) * tf.shape(params)[1], 0), 2) + ind
    ind3 = tf.reshape(ind2, [-1, tf.shape(ind2)[2]])
    sparse_ind = tf.stack([tf.repeat(tf.range(tf.shape(ind3)[0]), tf.shape(ind3)[1]), tf.reshape(ind3, [-1])], axis=1)
    params_reshaped = tf.reshape(params, [-1, tf.shape(params)[2]])
    sparse_mat = tf.sparse.SparseTensor(tf.cast(sparse_ind, tf.int64), tf.reshape(coef, [-1]), [tf.shape(ind3)[0], tf.shape(params_reshaped)[0]])
    out = tf.sparse.sparse_dense_matmul(sparse_mat, params_reshaped)
    out_reshaped = tf.reshape(out, [tf.shape(A)[0], tf.shape(A)[1], tf.shape(params)[2]])
    return out_reshaped


class HighOrderActivation(tf.keras.layers.Layer):
    def __init__(self, arity, out_dim, l1_pen_coef=0.0, l2_pen_coef=0.0):
        super().__init__()
        self.arity = arity
        self.out_dim = out_dim
        self.l1_pen_coef = tf.Variable(l1_pen_coef, trainable=False)
        self.l2_pen_coef = tf.Variable(l2_pen_coef, trainable=False)

    def build(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[2] == self.arity
        self.input_dim = input_shape[1]
        self.params = tf.Variable(tf.random.normal([self.input_dim, 2 ** self.arity, self.out_dim]))

    def penalty(self):
        l1_terms = []
        l2_terms = []
        for i in range(self.arity):
            shape0 = 2 ** i
            shape1 = 2 ** (self.arity - i - 1)
            params_rs = tf.reshape(self.params, [self.input_dim, shape0, 2, shape1, self.out_dim])
            diff = params_rs[:, :, 0, :, :] - params_rs[:, :, 1, :, :]
            l1_terms.append(tf.reduce_mean(tf.abs(diff)))
            l2_terms.append(tf.reduce_mean(diff ** 2))
        return sum(l1_terms) * (self.l1_pen_coef / len(l1_terms)) + sum(l2_terms) * (self.l2_pen_coef / len(l2_terms))

    def call(self, X, training=None):
        assert len(X.shape) == 3
        assert X.shape[1] == self.input_dim
        assert X.shape[2] == self.arity
        if training:
            self.add_loss(self.penalty())
        # return fast_high_order_act(X, self.params)
        out = sparsemm_high_order_act(X, self.params)
        return tf.reshape(out, [tf.shape(X)[0], self.input_dim, self.out_dim])



class BatchHighOrderActivation(tf.keras.layers.Layer):
    def __init__(self, arity, out_dim, l1_pen_coef=0.0, l2_pen_coef=0.0):
        super().__init__()
        self.arity = arity
        self.out_dim = out_dim
        self.l1_pen_coef = tf.Variable(l1_pen_coef, trainable=False)
        self.l2_pen_coef = tf.Variable(l2_pen_coef, trainable=False)

    def build(self, input_shape):
        assert input_shape[-1] == self.arity
        self.input_dim = input_shape[-2]
        self.params = tf.Variable(tf.random.normal([self.input_dim, 2 ** self.arity, self.out_dim]))

    def call(self, X, training=None):
        if training:
            self.add_loss(self.penalty())
        X1 = tf.reshape(X, [-1, self.input_dim, self.arity])
        # Y1 = fast_high_order_act(X1, self.params)
        Y1 = sparsemm_high_order_act(X1, self.params)
        Y_shape = list(X.shape)
        Y_shape[0] = -1
        Y_shape[-1] = self.out_dim
        Y = tf.reshape(Y1, tf.constant(Y_shape))
        return Y

    def penalty(self):
        l1_terms = []
        l2_terms = []
        for i in range(self.arity):
            shape0 = 2 ** i
            shape1 = 2 ** (self.arity - i - 1)
            params_rs = tf.reshape(self.params, [self.input_dim, shape0, 2, shape1, self.out_dim])
            diff = params_rs[:, :, 0, :, :] - params_rs[:, :, 1, :, :]
            l1_terms.append(tf.reduce_mean(tf.abs(diff)))
            l2_terms.append(tf.reduce_mean(diff ** 2))
        return sum(l1_terms) * (self.l1_pen_coef / len(l1_terms)) + sum(l2_terms) * (self.l2_pen_coef / len(l2_terms))


# m = 4
# n = 3
# k = 2
# params = tf.random.normal([k, 2 ** n, 5])
# A = tf.random.normal([m, k, n])
# out = sparsemm_high_order_act(A, params)
# out2 = fast_high_order_act(A, params)
# print(out - out2)