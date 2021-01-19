import tensorflow as tf
import logging
import math
from enwik_step import train_step

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("enwik.log"),
                              logging.StreamHandler()]
)

data_bytes = open('data/enwik8', 'rb').read()
# data = tf.constant(enwik8_bytes)
raw_data = tf.io.decode_raw(data_bytes, tf.uint8)
unique_result = tf.unique(raw_data)
data = unique_result.idx
VOCAB_SIZE = unique_result.y.shape[0]

# uc = tf.unique_with_counts(data)
# p = uc.count / tf.reduce_sum(uc.count)
# tf.reduce_sum(tf.cast(uc.count, tf.float64) * -tf.math.log(p)) / tf.cast(tf.math.log(2.0), dtype=tf.float64) / 8.0

DATA_SIZE = data.shape[0]
BLOCK_SIZE = 256
NUM_BLOCKS = (DATA_SIZE - 1) // BLOCK_SIZE
EMBEDDING_DIM = 64
LSTM_DIM = 64

data_in = data[:-1]
data_out = data[1:]
data_in_blocked = tf.reshape(data_in[:(NUM_BLOCKS * BLOCK_SIZE)], [NUM_BLOCKS, BLOCK_SIZE])
data_out_blocked = tf.reshape(data_out[:(NUM_BLOCKS * BLOCK_SIZE)], [NUM_BLOCKS, BLOCK_SIZE])

dataset = tf.data.Dataset.from_tensor_slices((data_in_blocked, data_out_blocked)).batch(512)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    tf.keras.layers.LSTM(LSTM_DIM, return_sequences=True),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(VOCAB_SIZE)
])
model.summary(print_fn=logging.info)

LEARNING_RATE = 0.01
optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9)
optimizer._create_all_weights(model.trainable_variables)
logging.info(optimizer.get_config())

loss_obj = tf.losses.SparseCategoricalCrossentropy()
train_loss = tf.keras.metrics.Mean(name='train_loss')

NUM_EPOCHS = 1000
logging.info("Starting training")
epoch = 1
for _ in range(NUM_EPOCHS):
    train_loss.reset_states()
    for batch_in, batch_out in dataset:
        # with tf.GradientTape() as tape:
        #     pred = model(batch_in)
        #     loss = loss_obj(batch_out, pred) / math.log(2)
        # gradients = tape.gradient(loss, model.trainable_variables)
        # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # train_loss(loss)
        # # logging.info(loss)
        optimizer.learning_rate.assign(LEARNING_RATE / math.sqrt(epoch))
        train_step(batch_in, batch_out, model, loss_obj, train_loss, optimizer)
    logging.info("{}: loss={:.6f}, lr={:.6f}".format(epoch, train_loss.result() / 8, float(optimizer.learning_rate)))
    epoch += 1
