import tensorflow as tf
import numpy as np
from custom_model import OrthogonalButterfly

k = 8
N = 2 ** 16
batch_size = N // 8
width_pow = 6
depth = 32
dtype = tf.float32

np.random.seed(0)
A = np.random.normal(size=[k, k])
Q = np.linalg.svd(A)[0]
if np.linalg.det(Q) < 0:
    Q[0, :] = -Q[0, :]
Q = tf.constant(Q, dtype=dtype)

def make_data(N):
    X = tf.random.normal([N, k, k], dtype=dtype)
    Y = tf.matmul(X, Q)
    return X, Y

X_train, Y_train = make_data(N)
X_test, Y_test = make_data(N // 8)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensors((X_test, Y_test))

model = tf.keras.Sequential([
    tf.keras.Input([k, k]),
    # tf.keras.layers.Permute([1, 2]),
    tf.keras.layers.Reshape([k * k]),
    OrthogonalButterfly(width_pow, k * k, depth),
    tf.keras.layers.Reshape([k, k]),
    tf.keras.layers.Permute([1, 2]),
])
model.summary()

# loss_fn = tf.losses.MeanSquaredError()
loss_fn = tf.losses.MeanAbsoluteError()


model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=10.0, momentum=0.95),
              # tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.95),
              loss=loss_fn)
model.fit(train_ds, validation_data=test_ds, epochs=1000)


# for x in tf.sort(tf.reshape(model.layers[1].params, [-1])).numpy().tolist():
#     print(x)
