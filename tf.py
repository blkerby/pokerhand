import tensorflow as tf
import tensorflow_addons as tfa
import logging
import importlib
import custom_model
from custom_model import train_step, test_step, L1BallConstraint, Scaling, TropicalDense, LeakyReLU, RandomElasticDistortion, NoisyDense, L1Dense, L1BallDense
from adabelief_tf import AdaBeliefOptimizer

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("tf.log"),
                              logging.StreamHandler()]
)


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# x_train, x_test = (x_train - 127.5) / 127.5, (x_test - 127.5) / 127.5
# train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# x_train = x_train[..., tf.newaxis].astype("float32")
# x_test = x_test[..., tf.newaxis].astype("float32")

# train_ds = tf.data.Dataset.from_tensor_slices(
#     (x_train, y_train)).shuffle(10000).batch(256)

def make_dataset(X, y):
    return tf.data.Dataset.from_tensor_slices(
        (tf.cast(X[..., tf.newaxis] / 255.0, tf.float32),
         tf.cast(y, tf.int64)))


train_ds = make_dataset(x_train, y_train).shuffle(65536).batch(512)

test_ds = make_dataset(x_test, y_test).batch(2048)


def make_model(inp):
    # bias_scale = 0.1
    # slope_scale = 1.0
    # init_slope = 1.0
    # model = tf.keras.Sequential([
    #     tf.keras.layers.experimental.preprocessing.RandomRotation(0.05, fill_mode='constant'),
    #     tf.keras.layers.experimental.preprocessing.RandomZoom(0.1, fill_mode='constant'),
    #     RandomElasticDistortion(3, 0.5),
    #     # tf.keras.layers.experimental.preprocessing.RandomTranslation(0.05, 0.05, fill_mode='constant'),
    #     # tf.keras.layers.Dropout(0.1),
    #     # tf.keras.layers.Conv2D(32, [5, 5], strides=2, activation='relu'),
    #     # tf.keras.layers.Conv2D(32, [5, 5], strides=2),
    #     tf.keras.layers.Conv2D(32, [5, 5]),
    #     # tf.keras.layers.Dropout(0.1),
    #     tf.keras.layers.BatchNormalization(momentum=0.0),
    #     LeakyReLU(bias_scale, slope_scale, init_slope),
    #     # tf.keras.layers.PReLU(),
    #     # tf.keras.layers.Activation(tf.nn.relu),
    #     # tf.keras.layers.Activation(tf.math.atan),
    #     tf.keras.layers.MaxPool2D(),
    #     # tf.keras.layers.Dropout(0.1),
    #     # tf.keras.layers.Conv2D(32, [3, 3], strides=2, activation='relu'),
    #     # tf.keras.layers.Conv2D(32, [3, 3], strides=2),
    #     tf.keras.layers.Conv2D(32, [3, 3]),
    #     # tf.keras.layers.Dropout(0.1),
    #     tf.keras.layers.BatchNormalization(momentum=0.0),
    #     # tf.keras.layers.PReLU(),
    #     # tf.keras.layers.Activation(tf.nn.relu),
    #     # tf.keras.layers.Activation(tf.math.atan),
    #     LeakyReLU(bias_scale, slope_scale, init_slope),
    #     tf.keras.layers.MaxPool2D(),
    #     # tf.keras.layers.Conv2D(16, [3, 3], strides=2, activation='relu'),
    #     # tf.keras.layers.Dense(32, activation='relu'),
    #     # tf.keras.layers.DepthwiseConv2D([3, 3], strides=2, depth_multiplier=1, activation='relu'),
    #     # tf.keras.layers.Conv2D(32, [3, 3], strides=2, activation='relu'),
    #     # tfa.layers.Maxout(16),
    #     # tf.keras.layers.Conv2D(32, [4, 4], activation='relu', kernel_constraint=L1BallConstraint(axes=[0, 1, 2], num_iters=6, memory=False)),
    #     # tf.keras.layers.BatchNormalization(momentum=0.0, renorm=True),
    #     # tf.keras.layers.Dropout(0.1),
    #     # tf.keras.layers.Conv2D(32, [5, 5], strides=2, activation='relu'),
    #     # tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    #     # tfa.layers.Maxout(16),
    #     # tf.keras.layers.Conv2D(32, [5, 5], strides=2, activation='relu'),
    #     # tf.keras.layers.Conv2D(32, [5, 5], activation='relu', kernel_constraint=L1BallConstraint(axes=[0, 1, 2], num_iters=6, memory=False)),
    #     # tf.keras.layers.MaxPool2D(),
    #     # tf.keras.layers.BatchNormalization(momentum=0.0, renorm=True),
    #     # tf.keras.layers.GlobalMaxPool2D(),
    #     # tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
    #     tf.keras.layers.Flatten(),
    #     # tf.keras.layers.Dropout(0.1),
    #     # tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(256),
    #     tf.keras.layers.BatchNormalization(momentum=0.0),
    #     LeakyReLU(bias_scale, slope_scale, init_slope),
    #     tf.keras.layers.Dense(256),
    #     tf.keras.layers.BatchNormalization(momentum=0.0),
    #     LeakyReLU(bias_scale, slope_scale, init_slope),
    #     tf.keras.layers.Dense(10),
    #     # tf.keras.layers.Dense(10, kernel_constraint=L1BallConstraint(axes=[0], num_iters=6, memory=False)),
    #     # tf.keras.layers.BatchNormalization(momentum=0.0, renorm=True),
    #     # Scaling(factor=2000.0),
    # ])(inp)
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(256, activation='relu'),
    #     # tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(256, activation='relu'),
    #     # tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(256, activation='relu'),
    #     # tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(10),
    # ])
    # sigma = 0.0
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Flatten(),
    #     NoisyDense(sigma, 256, activation='relu'),
    #     # tf.keras.layers.Dropout(0.5),
    #     NoisyDense(sigma, 256, activation='relu'),
    #     # tf.keras.layers.Dropout(0.5),
    #     NoisyDense(sigma, 256, activation='relu'),
    #     # tf.keras.layers.Dropout(0.5),
    #     tf.keras.layers.Dense(10),
    #     # NoisyDense(sigma, 10),
    # ])
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Flatten(),
    #     L1Dense(256),
    #     tf.keras.layers.ReLU(),
    #     L1Dense(256),
    #     tf.keras.layers.ReLU(),
    #     L1Dense(256),
    #     tf.keras.layers.ReLU(),
    #     L1Dense(10),
    #     Scaling(200.0),
    #     # tf.keras.layers.Dense(10),
    # ])
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        L1BallDense(256),
        # tfa.layers.Maxout(128),
        tf.keras.layers.ReLU(),
        L1BallDense(256),
        # tfa.layers.Maxout(128),
        tf.keras.layers.ReLU(),
        L1BallDense(256),
        # tfa.layers.Maxout(128),
        # tf.keras.layers.ReLU(),
        # L1BallDense(256),
        # tfa.layers.Maxout(128),
        tf.keras.layers.ReLU(),
        L1BallDense(10),
        # tf.keras.layers.Dense(10),
        Scaling(40.0, 5.0),
    ])
    return model(inp)


ensemble_size = 2
inp = tf.keras.Input(shape=(28, 28, 1))
raw_base_models = [make_model(inp) for _ in range(ensemble_size)]
base_models = [tf.keras.Model(inputs=inp, outputs=m) for m in raw_base_models]
if ensemble_size > 1:
    ensemble_out = tf.keras.layers.Average()(raw_base_models)
else:
    ensemble_out = raw_base_models[0]
model = tf.keras.Model(inputs=inp, outputs=ensemble_out)
model.summary(print_fn=logging.info)
base_models[0].summary(print_fn=logging.info)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.995, beta_2=0.995, epsilon=1e-07, amsgrad=False)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0015, beta_1=0.98, beta_2=0.99, epsilon=1e-07, amsgrad=False)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.003, beta_1=0.98, beta_2=0.99, epsilon=1e-07, amsgrad=False)
# logging.info(optimizer.get_config())
# optimizer = AdaBeliefOptimizer(learning_rate=0.0005, beta1=0.995, beta2=0.995, epsilon=1e-07, amsgrad=False)
# optimizer = AdaBeliefOptimizer(learning_rate=0.0005, beta_1=0.995, beta_2=0.995)

# optimizer._create_all_weights(model.trainable_variables)
optimizers = [tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.99, beta_2=0.99, epsilon=1e-07, amsgrad=False)
              for _ in range(ensemble_size)]
for i, optimizer in enumerate(optimizers):
    optimizer._create_all_weights(base_models[i].trainable_variables)
logging.info(optimizers[0].get_config())

EPOCHS = 1000
# ema_beta = 0.995
# ema_beta = 0.0

cumul_preds = tf.zeros([len(y_test)] + list(model.output_shape[1:]))

preds_history = []
# optimizer.learning_rate = 0.0005
for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    shadow_vars = [v.value() * 0 for v in model.variables]
    shadow_wt = 0.0

    for images, labels in train_ds:
        # shadow_vars = [s * ema_beta + v.value() * (1 - ema_beta) for s, v in zip(shadow_vars, model.variables)]
        # shadow_wt = shadow_wt * ema_beta + (1 - ema_beta)
        shadow_vars = [s + v.value() for s, v in zip(shadow_vars, model.variables)]
        shadow_wt = shadow_wt + 1.0
        # train_step(images, labels, model, optimizer, loss_fn, train_loss, train_accuracy)
        for i, m in enumerate(base_models):
            train_step(images, labels, m, optimizers[i], loss_fn, train_loss, train_accuracy)

    saved_vars = [v.value() for v in model.variables]
    for s, v in zip(shadow_vars, model.variables):
        v.assign(s / shadow_wt)

    test_preds = []
    for test_images, test_labels in test_ds:
        test_preds.append(test_step(test_images, model))
    cumul_preds = cumul_preds + tf.concat(test_preds, axis=0)
    if len(preds_history) == 0:
        preds_history.append(tf.zeros_like(cumul_preds))
    preds_history.append(cumul_preds)

    ind_start = (len(preds_history) - 1) // 2
    average_preds = (cumul_preds - preds_history[ind_start]) / (len(preds_history) - 1 - ind_start)
    t_loss = loss_fn(y_test, average_preds)

    test_loss(t_loss)
    test_accuracy(y_test, average_preds)


    for s, v in zip(saved_vars, model.variables):
        v.assign(s)

    logging.info(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}, '
        # f'Scale: {tf.reduce_mean(tf.abs(base_models[0].layers[1].layers[-1].scale) * base_models[0].layers[1].layers[-1].factor)}'
        f'Scale: {tf.reduce_mean(tf.exp(base_models[0].layers[1].layers[-1].log_scale * base_models[0].layers[1].layers[-1].factor))}'
    )

#
# model.compile(optimizer=avg_opt,
#               loss=loss_fn,
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=10)
#
# model.evaluate(x_test,  y_test, verbose=2)

# tf.reduce_sum(tf.abs(model.layers[-3].kernel), axis=1)

# model.save("test_model")
