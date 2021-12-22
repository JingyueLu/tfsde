import math
import numpy as np
import tensorflow as tf


def cheap_stack(tensors, axis):
    if len(tensors) == 1:
        return tf.expand_dims(tensors[0], axis)
    else:
        return tf.stack(tensors, axis=axis)



def validate_input_path(x, t):
    if not x.dtype.is_floating:
        raise ValueError("X must both be floating point.")

    if x.ndim < 2:
        raise ValueError("X must have at least two dimensions, corresponding to time and channels. It instead has "
                         "shape {}.".format(tuple(x.shape)))

    if t is None:
        t = tf.linspace(0, x.shape[-2] - 1, x.shape[-2])
        t = tf.cast(t, x.dtype)

    if not t.dtype.is_floating:
        raise ValueError("t must both be floating point.")
    if len(t.shape) != 1:
        raise ValueError("t must be one dimensional. It instead has shape {}.".format(tuple(t.shape)))
    prev_t_i = -math.inf
    for t_i in t:
        if t_i <= prev_t_i:
            raise ValueError("t must be monotonically increasing.")
        prev_t_i = t_i

    if x.shape[-2] != t.shape[0]:
        raise ValueError("The time dimension of X must equal the length of t. X has shape {} and t has shape {}, "
                         "corresponding to time dimensions of {} and {} respectively."
                         .format(tuple(x.shape), tuple(t.shape), x.shape[-2], t.shape[0]))

    if t.shape[0] < 2:
        raise ValueError("Must have a time dimension of size at least 2. It instead has shape {}, corresponding to a "
                         "time dimension of size {}.".format(tuple(t.shape), t.shape[0]))

    return t




