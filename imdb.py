# Based on https://www.tensorflow.org/tutorials/text/word_embeddings
import io
import os
import re
import shutil
import string
import tensorflow as tf

from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
#
# dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
#                                     untar=True, cache_dir='.',
#                                     cache_subdir='')


dataset_dir = 'aclImdb'
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

batch_size = 1024
train_ds = tf.keras.preprocessing.text_dataset_from_directory(train_dir, batch_size=batch_size)
test_ds = tf.keras.preprocessing.text_dataset_from_directory(test_dir, batch_size=batch_size)


# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), '')


vocab_size = 1000
sequence_length = 500
embedding_dim = 2

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)
vectorize_layer.adapt(train_ds.map(lambda x, y: x))

preprocessed_train_ds = train_ds.map(lambda x, y: (vectorize_layer(x), y)).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
preprocessed_test_ds = test_ds.map(lambda x, y: (vectorize_layer(x), y)).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


core_model = Sequential([
    tf.keras.layers.Input([sequence_length], dtype=tf.int32),
    Embedding(vocab_size, embedding_dim, name="embedding"),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1)
])

full_model = Sequential([
    tf.keras.layers.Input([], dtype=tf.string),
    vectorize_layer,
    core_model
])

core_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.002, beta_1=0.9),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


core_model.fit(
    preprocessed_train_ds,
    validation_data=preprocessed_test_ds,  # val_ds,
    epochs=100)
# callbacks=[tensorboard_callback])
