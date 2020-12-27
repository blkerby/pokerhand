from typing import Tuple, Union, List
import tensorflow as tf



@tf.function
def use_model_train(images, model):
    return model(images, training=True)


@tf.function
def use_model_test(images, model):
    return model(images, training=False)


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


# class SimplexConstraint(tf.keras.constraints.Constraint):
#     def __init__(self, axes, num_iters: int, memory: bool):
#         self.axes = axes
#         self.num_iters = num_iters
#         self.memory = memory
#         self.t = None
#
#     def __call__(self, w):
#         if self.t is None:
#             t_shape = [n for i, n in enumerate(w.shape.as_list()) if i not in self.axes]
#             t = tf.zeros(t_shape)
#         else:
#             t = self.t
#         x, t = approx_simplex_projection(w, t, self.axes, self.num_iters)
#         if self.memory:
#             self.t = t
#         return x



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


class Scaling(tf.keras.layers.Layer):
    def __init__(self, factor: float, init_scale: float = 1.0):
        super().__init__()
        self.factor = factor
        self.init_scale = init_scale

    def build(self, input_shape):
        self.scale = tf.Variable(tf.fill(input_shape[1:], self.init_scale / self.factor))

    def call(self, X):
        return (self.scale * self.factor) * X


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


class LeakyReLU(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.slope_left = tf.Variable(tf.zeros(input_shape[1:]))
        self.slope_right = tf.Variable(tf.ones(input_shape[1:]))

    def call(self, X):
        return tf.where(X >= 0, X * self.slope_right, X * self.slope_left)


# def approx_simplex_projection_fast_backprop(c: tf.Tensor, t: tf.Tensor, axes: List[int], num_iters: int) -> Tuple[tf.Tensor, tf.Tensor]:
#     t_expanded_shape = [1 if i in axes else n for i, n in enumerate(c.shape.as_list())]
#     x = tf.maximum(c + tf.reshape(t, t_expanded_shape), tf.zeros([]))
#     for _ in range(num_iters - 1):
#         x, t = simplex_projection_iteration(c, x, t, axes, t_expanded_shape)
#     x = tf.stop_gradient(x)
#     t = tf.stop_gradient(t)
#     x, t = simplex_projection_iteration(c, x, t, axes, t_expanded_shape)
#     return x, t



# c = tf.constant([1.0, -1.8, 2.1, -2.3])
# t = tf.constant(0.0)
# # x1, t1 = approx_simplex_projection(c, t, [0], 3)
# x1, t1 = approx_l1_ball_projection(c, t, [0], 3)
# print(x1)


# with tf.GradientTape() as tape:
#     tape.watch(c)
#     x1, t1 = approx_simplex_projection_fast_backprop(c, t, [0], 3)
# J = tape.jacobian(x1, c)
# print(J)
# # print(fast_approx_simplex_projection(c, t, 0, 1))