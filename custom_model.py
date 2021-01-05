from typing import Tuple, Union, List
import tensorflow as tf
import tensorflow_probability as tfp
import math

#
# @tf.function
# def use_model_train(images, model):
#     return model(images, training=True)
#
#
# @tf.function
# def use_model_test(images, model):
#     return model(images, training=False)


@tf.function
def train_step(images, labels, model, optimizer, loss_fn, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, model):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    return predictions




# def approx_simplex_projection(x: torch.tensor, dim: int, num_iters: int) -> torch.tensor:
#     mask = torch.ones(list(x.shape), dtype=x.dtype, device=x.device)
#     with torch.no_grad():
#         for i in range(num_iters - 1):
#             n_act = torch.sum(mask, dim=dim)
#             x_sum = torch.sum(x * mask, dim=dim)
#             t = (x_sum - 1.0) / n_act
#             x1 = x - t.unsqueeze(dim=dim)
#             mask = (x1 >= 0).to(x.dtype)
#         n_act = torch.sum(mask, dim=dim)
#     x_sum = torch.sum(x * mask, dim=dim)
#     t = (x_sum - 1.0) / n_act
#     x1 = torch.clamp(x - t.unsqueeze(dim=dim), min=0.0)
#     # logging.info(torch.mean(torch.sum(x1, dim=1)))
#     return x1  #/ torch.sum(torch.abs(x1), dim=dim).unsqueeze(dim=dim)
#

def simplex_projection_iteration(c: tf.Tensor, x: tf.Tensor, t: tf.Tensor, axis: Union[int, List[int]],
                                 t_expanded_shape: List[int]) -> Tuple[tf.Tensor, tf.Tensor]:
    d = tf.maximum(tf.cast(tf.math.count_nonzero(x, axis), x.dtype), tf.ones([]))
    s = tf.reduce_sum(x, axis)
    t = t + (tf.ones_like(s) - s) / d
    x = tf.maximum(c + tf.reshape(t, t_expanded_shape), tf.zeros([]))
    return x, t


def approx_simplex_projection(c: tf.Tensor, t: tf.Tensor, axes: List[int], num_iters: int) -> Tuple[tf.Tensor, tf.Tensor]:
    t_expanded_shape = [1 if i in axes else n for i, n in enumerate(c.shape.as_list())]
    x = tf.maximum(c + tf.reshape(t, t_expanded_shape), tf.zeros([]))
    for _ in range(num_iters):
        x, t = simplex_projection_iteration(c, x, t, axes, t_expanded_shape)
    return x, t


def approx_l1_ball_projection(c: tf.Tensor, t: tf.Tensor, axes: List[int], num_iters: int) -> Tuple[tf.Tensor, tf.Tensor]:
    cs = tf.sign(c)
    ca = tf.abs(c)
    sca = tf.reduce_sum(ca, axes, keepdims=True)
    x, t = approx_simplex_projection(ca, t, axes, num_iters)
    return tf.where(sca >= 1.0, x * cs, c), t


class SimplexConstraint(tf.keras.constraints.Constraint):
    def __init__(self, axes, num_iters: int, memory: bool):
        self.axes = axes
        self.num_iters = num_iters
        self.memory = memory
        self.t = None

    def __call__(self, w):
        if self.t is None:
            t_shape = [n for i, n in enumerate(w.shape.as_list()) if i not in self.axes]
            t = tf.zeros(t_shape)
        else:
            t = self.t
        x, t = approx_simplex_projection(w, t, self.axes, self.num_iters)
        if self.memory:
            self.t = t
        return x


class BoxConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, X):
        return tf.clip_by_value(X, self.min_val, self.max_val)



class L1BallConstraint(tf.keras.constraints.Constraint):
    def __init__(self, axes, num_iters: int, memory: bool):
        self.axes = axes
        self.num_iters = num_iters
        self.memory = memory
        self.t = None

    def __call__(self, w):
        if self.t is None:
            t_shape = [n for i, n in enumerate(w.shape.as_list()) if i not in self.axes]
            t = tf.zeros(t_shape)
        else:
            t = self.t
        x, t = approx_l1_ball_projection(w, t, self.axes, self.num_iters)
        if self.memory:
            self.t = t
        return x


class L1Dense(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        shape = [2, self.input_dim, self.num_outputs]
        x = tf.math.log(tf.random.uniform(shape))
        x = x / tf.reduce_sum(x, axis=[0, 1], keepdims=True)
        self.kernel_pos_neg = tf.Variable(x, constraint=SimplexConstraint([0, 1], 8, False))
        self.bias = tf.Variable(tf.zeros([self.num_outputs]))

    def call(self, X):
        kernel = self.kernel_pos_neg[0, :, :] - self.kernel_pos_neg[1, :, :]
        # bias = tf.reduce_sum(self.kernel_pos_neg[1, :, :], axis=0)
        return tf.matmul(X, kernel) + tf.expand_dims(self.bias, 0)



class L1BallDense(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        shape = [self.input_dim, self.num_outputs]
        x = tf.math.log(tf.random.uniform(shape))
        x = x / tf.reduce_sum(x, axis=[0], keepdims=True)
        self.kernel = tf.Variable(x, constraint=L1BallConstraint([0], 8, False))
        self.bias = tf.Variable(tf.zeros([self.num_outputs]))

    def call(self, X):
        return tf.matmul(X, self.kernel) + tf.expand_dims(self.bias, 0)


class BetaActivation(tf.keras.layers.Layer):
    def __init__(self, init_val):
        super().__init__()
        self.init_val = init_val

    def build(self, input_shape):
        self.alpha0 = tf.Variable(tf.fill(input_shape[1:], self.init_val))
        self.beta0 = tf.Variable(tf.fill(input_shape[1:], self.init_val))

    def call(self, X):
        # alpha = tf.sqrt(1 + self.alpha0 ** 2) + self.alpha0
        # beta = tf.sqrt(1 + self.beta0 ** 2) + self.beta0
        alpha = tf.exp(self.alpha0)
        beta = tf.exp(self.beta0)
        # alpha_broadcast = tf.broadcast_to(tf.expand_dims(alpha, 0), X.shape)
        # beta_broadcast = tf.broadcast_to(tf.expand_dims(beta, 0), X.shape)
        # alpha_broadcast = tf.expand_dims(alpha, 0) + tf.zeros_like(X)
        # beta_broadcast = tf.expand_dims(beta, 0) + tf.zeros_like(X)
        dist = tfp.distributions.Kumaraswamy(alpha, beta)
        return dist.cdf(tf.clip_by_value(X, 0.0, 1.0 - 1e-6))
        # return tf.math.betainc(alpha_broadcast, beta_broadcast, X)
        # dist = tfp.distributions.Beta(alpha_broadcast, beta_broadcast)
        # return dist.cdf(X)


# d = L1Dense(2)
# bact = BetaActivation(0.0)
# x = tf.random.uniform([2, 2])
# y = d(x)
# bact(y)
# with tf.GradientTape() as tape:
#     tape.watch(x)
#     y = bact(x)
# print(tape.gradient(y, [x]))





class Scaling(tf.keras.layers.Layer):
    def __init__(self, factor: float, init_scale: float):
        super().__init__()
        self.factor = factor
        self.init_scale = init_scale

    def build(self, input_shape):
        # self.scale = tf.Variable(tf.fill(input_shape[1:], self.init_scale / self.factor))
        self.log_scale = tf.Variable(tf.zeros(input_shape[1:]))
        self.bias = tf.Variable(tf.zeros(input_shape[1:]))

    def call(self, X):
        # scale = tf.expand_dims(self.scale * self.factor, 0)
        scale = tf.expand_dims(tf.exp(self.log_scale * self.factor), 0)
        return scale * X + tf.expand_dims(self.bias, 0)


class TropicalDense(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        shape = [2] + input_shape[1:] + [self.num_outputs]
        self.kernel = tf.Variable(tf.random.normal(shape))

    def call(self, X):
        X1 = tf.expand_dims(tf.stack([X, -X], axis=1), -1)
        axis_list = list(range(1, len(X.shape) + 1))
        out = X1 + tf.expand_dims(self.kernel, 0)
        return tf.reduce_max(out, axis=axis_list)


class AbsoluteValue(tf.keras.layers.Layer):
    def __init__(self, bias_scale):
        super().__init__()
        self.bias_scale = bias_scale

    def build(self, input_shape):
        self.bias = tf.Variable(tf.zeros(input_shape[1:]))

    def call(self, X):
        X1 = X + tf.expand_dims(self.bias * self.bias_scale, 0)
        return tf.abs(X1)


class LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, bias_scale, slope_scale, init_slope):
        super().__init__()
        self.bias_scale = bias_scale
        self.slope_scale = slope_scale
        self.init_slope = init_slope

    def build(self, input_shape):
        self.slope_left = tf.Variable(tf.zeros(input_shape[1:]))
        self.slope_right = tf.Variable(tf.ones(input_shape[1:]) * self.init_slope / self.slope_scale)
        self.bias = tf.Variable(tf.zeros(input_shape[1:]))

    def call(self, X):
        X1 = X + tf.expand_dims(self.bias * self.bias_scale, 0)
        return tf.where(X1 >= 0, X1 * (self.slope_right * self.slope_scale), X1 * (self.slope_left * self.slope_scale))


class ConstrainedLeakyReLU(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.slope_left = tf.Variable(-tf.ones(input_shape[1:]), constraint=BoxConstraint(-1.0, 1.0))
        # self.slope_left = tf.Variable(tf.zeros(input_shape[1:]), constraint=BoxConstraint(-1.0, 1.0))
        self.slope_right = tf.Variable(tf.ones(input_shape[1:]), constraint=BoxConstraint(-1.0, 1.0))
        self.bias = tf.Variable(tf.zeros(input_shape[1:]))

    def call(self, X):
        X1 = X + tf.expand_dims(self.bias, 0)
        return tf.where(X1 >= 0, X1 * self.slope_right, X1 * self.slope_left)


class RevLeakyReLU(tf.keras.layers.Layer):
    def __init__(self, bias_scale, slope_scale, init_slope):
        super().__init__()
        self.bias_scale = bias_scale
        self.slope_scale = slope_scale
        self.init_slope = init_slope

    def build(self, input_shape):
        self.log_slope_left = tf.Variable(tf.random.normal(input_shape[1:]) * (self.init_slope / self.slope_scale))
        self.log_slope_right = tf.Variable(tf.random.normal(input_shape[1:]) * (self.init_slope / self.slope_scale))
        # self.log_slope_left = tf.Variable(tf.zeros(input_shape[1:]))
        # self.log_slope_right = tf.Variable(tf.zeros(input_shape[1:]))
        # self.log_slope_right = tf.Variable(tf.ones(input_shape[1:]) / self.slope_scale)
        self.bias = tf.Variable(tf.zeros(input_shape[1:]))

    def call(self, X):
        slope_left = tf.exp(self.log_slope_left * self.slope_scale)
        slope_right = tf.exp(self.log_slope_right * self.slope_scale)
        X1 = X + tf.expand_dims(self.bias * self.bias_scale, 0)
        return tf.where(X1 >= 0, X1 * slope_right, X1 * slope_left)

# def approx_simplex_projection_fast_backprop(c: tf.Tensor, t: tf.Tensor, axes: List[int], num_iters: int) -> Tuple[tf.Tensor, tf.Tensor]:
#     t_expanded_shape = [1 if i in axes else n for i, n in enumerate(c.shape.as_list())]
#     x = tf.maximum(c + tf.reshape(t, t_expanded_shape), tf.zeros([]))
#     for _ in range(num_iters - 1):
#         x, t = simplex_projection_iteration(c, x, t, axes, t_expanded_shape)
#     x = tf.stop_gradient(x)
#     t = tf.stop_gradient(t)
#     x, t = simplex_projection_iteration(c, x, t, axes, t_expanded_shape)
#     return x, t



# c = tf.constant([[0.2, 1.3], [1.7, 1.8]])
# t = tf.constant([0.0, 0.0])
# x1, t1 = approx_simplex_projection(c, t, [1], 3)
# # x1, t1 = approx_l1_ball_projection(c, t, [0], 3)
# print(x1)


# with tf.GradientTape() as tape:
#     tape.watch(c)
#     x1, t1 = approx_simplex_projection_fast_backprop(c, t, [0], 3)
# J = tape.jacobian(x1, c)
# print(J)
# # print(fast_approx_simplex_projection(c, t, 0, 1))


class OrthogonalButterfly(tf.keras.layers.Layer):
    def __init__(self, input_width_pow, output_width, depth):
        super().__init__()
        self.input_width_pow = input_width_pow
        self.input_width = 2 ** input_width_pow
        self.half_width = 2 ** (input_width_pow - 1)
        self.output_width = output_width
        self.depth = depth
        initial_params = tf.random.uniform([self.half_width, depth], -math.pi, math.pi)
        self.params = tf.Variable(initial_params)
        perm = []
        for i in range(self.input_width):
            if i % 2 == 0:
                perm.append(i // 2)
            else:
                perm.append(i // 2 + self.half_width)
        self.perm = tf.constant(perm)

    def build(self, input_shape):
        pass

    def call(self, X):
        assert X.dtype == self.dtype
        # if X.shape[0] is None:
        #     print(X)
        # X = tf.concat([X, tf.zeros([1, self.input_width - X.shape[1]])], axis=1)
        if self.input_width != X.shape[1]:
            X = tf.pad(X, tf.constant([[0, 0], [0, self.input_width - X.shape[1]]]))
        # X = tf.concat([X, tf.zeros([X.shape[0], self.input_width - X.shape[1]])], axis=1)
        for i in range(self.depth):
            X0 = X[:, :self.half_width]
            X1 = X[:, self.half_width:]
            theta = self.params[:, i]
            cos_theta = tf.cos(theta)
            sin_theta = tf.sin(theta)
            new_X0 = X0 * cos_theta + X1 * sin_theta
            new_X1 = X0 * -sin_theta + X1 * cos_theta
            X = tf.concat([new_X0, new_X1], axis=1)
            X = tf.gather(X, self.perm, axis=1)
        return X[:, :self.output_width]


class Permute(tf.keras.layers.Layer):
    def __init__(self, width):
        super().__init__()
        self.perm = tf.random.shuffle(tf.range(width))

    def call(self, X):
        return tf.gather(X, self.perm, axis=1)


def rand_smooth(m, n, deg, sigma):
    A_real = tf.random.normal([m, deg, deg]) * (sigma / math.sqrt(deg))
    A_imag = tf.random.normal([m, deg, deg]) * (sigma / math.sqrt(deg))
    A = tf.complex(A_real, A_imag)
    A = tf.pad(A, [[0, 0], [0, n - deg], [0, n - deg]])
    A = tf.cast(tf.math.real(tf.signal.fft2d(A)), dtype=tf.int32)
    return A


def elastic_distortion(images, deg, sigma):
    m = tf.shape(images)[0]
    n = tf.shape(images)[1]
    dx = rand_smooth(m, n, deg, sigma)
    dy = rand_smooth(m, n, deg, sigma)
    x = tf.expand_dims(tf.expand_dims(tf.range(n, dtype=tf.int32), 0), 2) + dx
    y = tf.expand_dims(tf.expand_dims(tf.range(n, dtype=tf.int32), 0), 1) + dy
    x = tf.clip_by_value(x, 0, n - 1)
    y = tf.clip_by_value(y, 0, n - 1)
    out = tf.gather_nd(images, tf.stack([x, y], axis=-1), batch_dims=1)
    return out

class RandomElasticDistortion(tf.keras.layers.Layer):
    def __init__(self, degree, sigma):
        super().__init__()
        self.degree = degree
        self.sigma = sigma

    def build(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == input_shape[2]
        self.image_size = input_shape[1]
        self.channels = input_shape[3]

    def call(self, X):
        X_split = tf.split(X, 2)
        distorted = elastic_distortion(X_split[0], self.degree, self.sigma)
        distorted = tf.reshape(distorted, tf.shape(X_split[0]))  # This doesn't actually reshape, just helps Tensorflow infer output shape
        return tf.concat([distorted, X_split[1]], axis=0)


class NoisyDense(tf.keras.layers.Dense):
    def __init__(self, sigma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma

    def call(self, X, training=None):
        clean_out = super().call(X)
        if training:
            noise_scale = tf.sqrt(tf.reduce_sum(X ** 2, axis=1) + 1.0) * self.sigma
            noise = tf.random.normal(tf.shape(clean_out)) * tf.expand_dims(noise_scale, 1)
            return clean_out + noise
        else:
            return clean_out
