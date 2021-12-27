import math
import tensorflow as tf
import warnings

from . import interpolation_base
from . import misc


_two_pi = 2 * math.pi
_inv_two_pi = 1 / _two_pi


def _linear_interpolation_coeffs_with_missing_values_scalar(t, x):
    # t and X both have shape (length,)

    not_nan = ~tf.math.is_nan(x)
    path_no_nan = tf.boolean_mask(x, not_nan)

    if path_no_nan.shape[0] == 0:
        # Every entry is a NaN, so we take a constant path with derivative zero, so return zero coefficients.
        return tf.zeros(x.shape[0], dtype=x.dtype)

    if path_no_nan.shape[0] == x.shape[0]:
        # Every entry is not-NaN, so just return.
        return x

    x = tf.Variable(x, trainable=False)
    # How to deal with missing values at the start or end of the time series? We impute an observation at the very start
    # equal to the first actual observation made, and impute an observation at the very end equal to the last actual
    # observation made, and then proceed as normal.
    if tf.math.is_nan(x[0]):
        x[0].assign(path_no_nan[0])
    if tf.math.is_nan(x[-1]):
        x[-1].assign(path_no_nan[-1])

    nan_indices = tf.boolean_mask(tf.range(x.shape[0]), tf.math.is_nan(x))

    if nan_indices.shape[0] == 0:
        # We only had missing values at the start or end
        return x

    prev_nan_index = nan_indices[0]
    prev_not_nan_index = prev_nan_index - 1
    prev_not_nan_indices = [prev_not_nan_index]
    for nan_index in nan_indices[1:]:
        if prev_nan_index != nan_index - 1:
            prev_not_nan_index = nan_index - 1
        prev_nan_index = nan_index
        prev_not_nan_indices.append(prev_not_nan_index)

    next_nan_index = nan_indices[-1]
    next_not_nan_index = next_nan_index + 1
    next_not_nan_indices = [next_not_nan_index]
    for nan_index in reversed(nan_indices[:-1]):
        if next_nan_index != nan_index + 1:
            next_not_nan_index = nan_index + 1
        next_nan_index = nan_index
        next_not_nan_indices.append(next_not_nan_index)
    next_not_nan_indices = reversed(next_not_nan_indices)
    for prev_not_nan_index, nan_index, next_not_nan_index in zip(prev_not_nan_indices,
                                                                 nan_indices,
                                                                 next_not_nan_indices):
        prev_stream = x[prev_not_nan_index]
        next_stream = x[next_not_nan_index]
        prev_time = t[prev_not_nan_index]
        next_time = t[next_not_nan_index]
        time = t[nan_index]
        ratio = (time - prev_time) / (next_time - prev_time)
        x[nan_index].assign( prev_stream + ratio * (next_stream - prev_stream) )

    return x+0  # convert tf.variable to tf.constant


def _linear_interpolation_coeffs_with_missing_values(t, x):
    if x.ndim == 1:
        # We have to break everything down to individual scalar paths because of the possibility of missing values
        # being different in different channels
        return _linear_interpolation_coeffs_with_missing_values_scalar(t, x)
    else:
        out_pieces = []
        for idx in range(x.shape[0]) :  # TODO: parallelise over this
            out = _linear_interpolation_coeffs_with_missing_values(t, x[idx])
            out_pieces.append(out)
            print(idx)
        return misc.cheap_stack(out_pieces, axis=0)



def linear_interpolation_coeffs(x, t=None):
    """Calculates the knots of the linear interpolation of the batch of controls given.

    Arguments:
        x: tensor of values, of shape (..., length, input_channels), where ... is some number of batch dimensions. This
            is interpreted as a (batch of) paths taking values in an input_channels-dimensional real vector space, with
            length-many observations. Missing values are supported, and should be represented as NaNs.
        t: Optional one dimensional tensor of times. Must be monotonically increasing. If not passed will default to
            tensor([0., 1., ..., length - 1]). If you are using neural CDEs then you **do not need to use this
            argument**. See the Further Documentation in README.md.

    Warning:
        If there are missing values then calling this function can be pretty slow. Make sure to cache the result, and
        don't call it on every forward pass, if at all possible.

    Returns:
        A tensor, which should in turn be passed to `tfcde.LinearInterpolation`.

    """

    t = misc.validate_input_path(x, t)

    if tf.math.reduce_any(tf.math.is_nan(x)):
        x = _linear_interpolation_coeffs_with_missing_values(t, tf.linalg.matrix_transpose(x))
        x = tf.linalg.matrix_transpose(x)
    return x


class LinearInterpolation(interpolation_base.InterpolationBase):
    """Calculates the linear interpolation to the batch of controls given. Also calculates its derivative."""

    def __init__(self, coeffs, t=None, **kwargs):
        """
        Arguments:
            coeffs: As returned by linear_interpolation_coeffs.
            t: As passed to linear_interpolation_coeffs. (If it was passed. If you are using neural CDEs then you **do
                not need to use this argument**. See the Further Documentation in README.md.)
        """
        super(LinearInterpolation, self).__init__(**kwargs)

        if t is None:
            t = tf.linspace(0, coeffs.shape[-2] - 1, coeffs.shape[-2])
            t = tf.cast(t, dtype=coeffs.dtype)

        #derivs = (coeffs[..., 1:, :] - coeffs[..., :-1, :]) / tf.expand_dims(t[1:] - t[:-1], axis = -1)

        self._t =  t
        self._coeffs = coeffs
        #self._derivs = derivs

    @property
    def grid_points(self):
        return self._t

    @property
    def interval(self):
        return tf.stack([self._t[0], self._t[-1]])

    def _interpret_t(self, t):
        maxlen = self._coeffs.shape[-2] - 2
        ## clamp because t may go outside of [t[0], t[-1]]; this is fine
        index = tf.clip_by_value(tf.raw_ops.Bucketize(input=t, boundaries=self._t.numpy().tolist()) - 1, 0, maxlen)
        fractional_part = t- self._t[index]
        return fractional_part, index

    def evaluate(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = tf.expand_dims(fractional_part, axis=-1)
        prev_coeff = self._coeffs[..., index, :]
        next_coeff = self._coeffs[..., index + 1, :]
        prev_t = self._t[index]
        next_t = self._t[index + 1]
        diff_t = next_t - prev_t
        return prev_coeff + fractional_part * (next_coeff - prev_coeff) / tf.expand_dims(diff_t, axis= -1)

    def derivative(self, t):
        fractional_part, index = self._interpret_t(t-1)
        deriv = (self._coeffs[..., index+1, :] - self._coeffs[..., index, :]) / (self._t[index+1] - self._t[index])
        return deriv
