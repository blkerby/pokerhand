import tensorflow as tf
import tensorflow_addons as tfa
import logging
from custom_model import train_step, test_step, Scaling, TropicalDense, LeakyReLU, RandomElasticDistortion, NoisyDense, \
    L1Dense, L1BallDense, L1BallConstraint, SimplexConstraint, L1PositiveDense, Bias
from high_order_act import HighOrderActivation, BatchHighOrderActivation
from adabelief_tf import AdaBeliefOptimizer

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("tf.log"),
                              logging.StreamHandler()]
)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def make_dataset(X, y):
    return tf.data.Dataset.from_tensor_slices(
        (tf.cast(X[..., tf.newaxis] / 255.0, tf.float32),
         tf.cast(y, tf.int64)))

BATCH_SIZE = 512
train_ds = make_dataset(x_train, y_train).shuffle(65536).batch(BATCH_SIZE)
test_ds = make_dataset(x_test, y_test).batch(BATCH_SIZE)


def make_model():
    # bias_scale = 0.1
    # slope_scale = 1.0
    # init_slope = 1.0
    act_l1_pen_coef = 0.0
    act_l2_pen_coef = 0.0
    pen_coef = 1e-5
    pen_exp = 10.0
    act_arity = 10
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(act_arity, [3, 3], strides=2, kernel_constraint=SimplexConstraint([0, 1, 2], 8, False)),
        tf.keras.layers.Reshape([13, 13, 1, act_arity]),
        BatchHighOrderActivation(act_arity, act_arity, l1_pen_coef=act_l1_pen_coef, l2_pen_coef=act_l2_pen_coef),
        tf.keras.layers.Reshape([13, 13, act_arity]),
        tf.keras.layers.Conv2D(act_arity, [3, 3], strides=2, kernel_constraint=SimplexConstraint([0, 1, 2], 8, False)),
        tf.keras.layers.Reshape([6, 6, 1, act_arity]),
        BatchHighOrderActivation(act_arity, act_arity, l1_pen_coef=act_l1_pen_coef, l2_pen_coef=act_l2_pen_coef),
        tf.keras.layers.Reshape([6, 6, act_arity]),
        tf.keras.layers.Flatten(),
        L1Dense(32 * act_arity, pen_coef=pen_coef, pen_exp=pen_exp),
        tf.keras.layers.Reshape([-1, act_arity]),
        HighOrderActivation(act_arity, act_arity, l1_pen_coef=act_l1_pen_coef, l2_pen_coef=act_l2_pen_coef),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10),
        # L1Dense(10, pen_coef=pen_coef, pen_exp=pen_exp),
    ])

    # # bias_scale = 0.1
    # # slope_scale = 1.0
    # # init_slope = 1.0
    # act_l2_pen_coef = 0.1
    # pen_coef = 1e-5
    # pen_exp = 10.0
    # act_arity = 32
    # model = tf.keras.Sequential([
    #     # tf.keras.layers.experimental.preprocessing.RandomRotation(0.05, fill_mode='constant'),
    #     # tf.keras.layers.experimental.preprocessing.RandomZoom(0.1, fill_mode='constant'),
    #     # RandomElasticDistortion(3, 0.5),
    #     tf.keras.layers.Conv2D(act_arity, [5, 5], strides=2),
    #     # tf.keras.layers.Reshape([12, 12, 1, act_arity]),
    #     # BatchHighOrderActivation(act_arity, act_arity),
    #     # tf.keras.layers.Reshape([12, 12, act_arity]),
    #     tf.keras.layers.ReLU(),
    #     tf.keras.layers.Conv2D(act_arity, [5, 5], strides=2),
    #     # tf.keras.layers.Reshape([4, 4, 1, act_arity]),
    #     # BatchHighOrderActivation(act_arity, act_arity),
    #     # tf.keras.layers.Reshape([4, 4, act_arity]),
    #     tf.keras.layers.ReLU(),
    #     # tf.keras.layers.MaxPool2D(),
    #     tf.keras.layers.Flatten(),
    #     L1Dense(act_arity * 16, pen_coef=pen_coef, pen_exp=pen_exp),
    #     # tf.keras.layers.Dense(act_arity * 16),
    #     # tf.keras.layers.Reshape([-1, act_arity]),
    #     # HighOrderActivation(act_arity, act_arity, l2_pen_coef=act_l2_pen_coef),
    #     # HighOrderActivation(act_arity, 10, l2_pen_coef=act_l2_pen_coef),
    #     tf.keras.layers.ReLU(),
    #     tf.keras.layers.Flatten(),
    #     # # tf.keras.layers.ReLU(),
    #     # L1Dense(128, pen_coef=pen_coef, pen_exp=pen_exp),
    #     # tf.keras.layers.Reshape([-1, 8]),
    #     # HighOrderActivation(8, 8, l2_pen_coef=act_l2_pen_coef),
    #     # # tf.keras.layers.ReLU(),
    #     # tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(10),
    #     # Bias(),
    # ])

    return model


ensemble_size = 1
inp = tf.keras.Input(shape=(28, 28, 1))
preprocessing = tf.keras.Sequential([
    # tf.keras.layers.experimental.preprocessing.RandomRotation(0.05, fill_mode='constant'),
    # tf.keras.layers.experimental.preprocessing.RandomZoom(0.1, fill_mode='constant'),
    # RandomElasticDistortion(3, 0.5),
    # tf.keras.layers.experimental.preprocessing.RandomRotation(0.03, fill_mode='constant'),
    # tf.keras.layers.experimental.preprocessing.RandomZoom(0.06, fill_mode='constant'),
    # RandomElasticDistortion(3, 0.3),
])
preprocessed_inp = preprocessing(inp)
raw_base_models = [make_model() for _ in range(ensemble_size)]
base_models = [m(preprocessed_inp) for m in raw_base_models]
if ensemble_size > 1:
    ensemble_out = tf.keras.layers.Average()(base_models)
else:
    ensemble_out = base_models[0]
model = tf.keras.Model(inputs=inp, outputs=ensemble_out)
model.summary(print_fn=logging.info)
raw_base_models[0].summary(print_fn=logging.info)

# with tf.GradientTape() as tape:
#     predictions = model(images, training=True)
#     loss = loss_fn(labels, predictions)
#     obj = loss + sum(model.losses)
# gradients = tape.gradient(obj, model.trainable_variables)
# tf.print(gradients[0])

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_obj = tf.keras.metrics.Mean(name='train_obj')
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


optimizers = [tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.95, beta_2=0.999, epsilon=1e-07, amsgrad=False)
              for _ in range(ensemble_size)]

# optimizers = [AdaBeliefOptimizer(learning_rate=0.005, beta_1=0.995, beta_2=0.999, rectify=False)
#               for _ in range(ensemble_size)]

# optimizers = [tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.99)
#               for _ in range(ensemble_size)]

for i, optimizer in enumerate(optimizers):
    optimizer._create_all_weights(raw_base_models[i].trainable_variables)
logging.info(optimizers[0].get_config())

EPOCHS = 5000
# ema_beta = 0.995
# ema_beta = 0.0

cumul_preds = tf.zeros([len(y_test)] + list(model.output_shape[1:]))

history_frac = 0.5
preds_history = []

# for optimizer in optimizers:
#     optimizer.learning_rate = 1e-8
#     optimizer.beta_1 = 0.999
#     optimizer.beta_1 = 0.9998
# for model in raw_base_models:
#     for layer in model.layers:
#         if hasattr(layer, "pen_coef"):
#             layer.pen_coef.assign(3e-4)
#             layer.pen_exp.assign(5.0)

for _ in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_obj.reset_states()
    train_accuracy.reset_states()

    # for m in raw_base_models:
    #     for layer in m.layers:
    #         if hasattr(layer, 'pen_coef'):
    #             layer.pen_coef.assign(5e-6 * epoch)

    shadow_vars = [v.value() * 0 for v in model.trainable_variables]
    shadow_wt = 0.0

    for images, labels in train_ds:
        # shadow_vars = [s * ema_beta + v.value() * (1 - ema_beta) for s, v in zip(shadow_vars, model.trainable_variables)]
        # shadow_wt = shadow_wt * ema_beta + (1 - ema_beta)
        shadow_vars = [s + v.value() for s, v in zip(shadow_vars, model.trainable_variables)]
        shadow_wt = shadow_wt + 1.0
        # train_step(images, labels, model, optimizer, loss_fn, train_loss, train_accuracy)
        prep_images = preprocessing(images)
        for i, m in enumerate(raw_base_models):
            train_step(prep_images, labels, m, optimizers[i], loss_fn, train_loss, train_obj, train_accuracy)
            # print(train_loss.result(), train_accuracy.result())

    saved_vars = [v.value() for v in model.trainable_variables]
    for s, v in zip(shadow_vars, model.trainable_variables):
        v.assign(s / shadow_wt)

    test_preds = []
    for test_images, test_labels in test_ds:
        test_preds.append(test_step(test_images, model))
    cumul_preds = cumul_preds + tf.concat(test_preds, axis=0)
    if len(preds_history) == 0:
        preds_history.append(tf.zeros_like(cumul_preds))
    preds_history.append(cumul_preds)

    ind_start = int((len(preds_history) - 1) * (1.0 - history_frac))
    average_preds = (cumul_preds - preds_history[ind_start]) / (len(preds_history) - 1 - ind_start)
    t_loss = loss_fn(y_test, average_preds)

    test_loss.reset_states()
    test_accuracy.reset_states()
    test_loss(t_loss)
    test_accuracy(y_test, average_preds)

    act_cnt = 0
    for v in model.trainable_variables:
        # if hasattr(v, 'constraint') and isinstance(v.constraint, L1BallConstraint):
        #     act_cnt += tf.math.count_nonzero(v)
        if hasattr(v, 'constraint') and isinstance(v.constraint, SimplexConstraint):
            act_cnt += tf.math.count_nonzero(v)

    epoch = len(preds_history) - 1
    logging.info(
        '{}: loss={:.5f}, obj={:.5f}, acc={:.5f}, test_loss={:.5f}, test_acc={:.5f}, act={}'.format(
            epoch, train_loss.result(), train_obj.result(), train_accuracy.result(),
            test_loss.result(), test_accuracy.result(), act_cnt))

    for s, v in zip(saved_vars, model.trainable_variables):
        v.assign(s)


    # logging.info(
    #     '{}: loss={:.5f}, obj={:.5f}, acc={:.5f}, test_loss={:.5f}, test_acc={:.5f}'.format(
    #         epoch + 1, train_loss.result(), train_obj.result(), train_accuracy.result(),
    #         test_loss.result(), test_accuracy.result()))


    # avg_scale = tf.reduce_mean(tf.abs(raw_base_models[0].layers[-1].scale) * raw_base_models[0].layers[-1].factor)
    # logging.info(
    #     '{}: loss={:.5f}, obj={:.5f}, acc={:.5f}, test_loss={:.5f}, test_acc={:.5f}, scale={:.5f}, act={}'.format(
    #         epoch + 1, train_loss.result(), train_obj.result(), train_accuracy.result(),
    #         test_loss.result(), test_accuracy.result(), avg_scale, act_cnt))

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
