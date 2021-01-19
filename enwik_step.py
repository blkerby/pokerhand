import math
import tensorflow as tf

@tf.function
def train_step(batch_in, batch_out, model, loss_obj, train_loss, optimizer):
    with tf.GradientTape() as tape:
        pred = model(batch_in)
        loss = loss_obj(batch_out, pred) / math.log(2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
