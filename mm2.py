import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from custom_model import OrthogonalButterfly

k = 4
r = 49
N = 2 ** 20
batch_size = 2 ** 16
num_batches = N // batch_size
dtype = tf.float32


def make_data(N):
    X1 = tf.random.normal([N, k, k], dtype=dtype)
    X2 = tf.random.normal([N, k, k], dtype=dtype)
    Y = tf.matmul(X1, X2)
    return X1, X2, Y


X1_train, X2_train, Y_train = make_data(N)
X1_test, X2_test, Y_test = make_data(N // 8)

train_ds = tf.data.Dataset.from_tensor_slices(((X1_train, X2_train), Y_train)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensors(((X1_test, X2_test), Y_test))

# class BilinearMap(tf.keras.layers.Layer):
#     def __init__(self, k):
#         super().__init__()
#         self.k = k
#
#     def call(self, X1, X2):


inp1 = tf.keras.Input([k, k])
inp2 = tf.keras.Input([k, k])
flatten1 = tf.keras.layers.Flatten()(inp1)
flatten2 = tf.keras.layers.Flatten()(inp2)
dense1 = tf.keras.layers.Dense(r, use_bias=False)(flatten1)
dense2 = tf.keras.layers.Dense(r, use_bias=False)(flatten2)
product = tf.keras.layers.Multiply()([dense1, dense2])
out_flat = tf.keras.layers.Dense(k * k, use_bias=False)(product)
out = tf.keras.layers.Reshape([k, k])(out_flat)
model = tf.keras.Model(inputs=[inp1, inp2], outputs=out)
model.summary()

loss_fn = tf.losses.MeanSquaredError()
# loss_fn = tf.losses.MeanAbsoluteError()

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.002, decay_steps=num_batches * 20, decay_rate=0.5, staircase=True, name=None
# )
lr_schedule = 0.0005
model.compile(
    # optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule,
    #                                   momentum=0.9),
    tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9),
    # tfa.optimizers.AdamW(learning_rate=lr_schedule, beta_1=0.9, weight_decay=0.001),
    loss=loss_fn)
model.fit(train_ds, validation_data=test_ds, epochs=100000)

# for x in tf.sort(tf.reshape(model.layers[1].params, [-1])).numpy().tolist():
#     print(x)
