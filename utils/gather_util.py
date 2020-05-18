import tensorflow as tf
import numpy as np

def custom_gather(a, b):
    unstacked_a = tf.unstack(a, axis=0)
    unstacked_b = tf.unstack(b, axis=0)
    gathered = [tf.gather(x, y) for x, y in zip(unstacked_a, unstacked_b)]
    return tf.stack(gathered, axis=0)
