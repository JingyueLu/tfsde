import tfsde
import tensorflow as tf
from tensorflow.keras import layers
import warnings


def _check_compatability_per_tensor_base(control_gradient, z0):
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("X.derivative did not return a tensor with the same number of batch dimensions as z0. "
                         "X.derivative returned shape {} (meaning {} batch dimensions), whilst z0 has shape {} "
                         "(meaning {} batch dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))


def _check_compatability_per_tensor_forward(control_gradient, system, z0):
    _check_compatability_per_tensor_base(control_gradient, z0)
    if system.shape[:-2] != z0.shape[:-1]:
        raise ValueError("func did not return a tensor with the same number of batch dimensions as z0. func returned "
                         "shape {} (meaning {} batch dimensions), whilst z0 has shape {} (meaning {} batch"
                         " dimensions)."
                         "".format(tuple(system.shape), tuple(system.shape[:-2]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    if system.shape[-2] != z0.shape[-1]:
        raise ValueError("func did not return a tensor with the same number of hidden channels as z0. func returned "
                         "shape {} (meaning {} channels), whilst z0 has shape {} (meaning {} channels)."
                         "".format(tuple(system.shape), system.shape[-2], tuple(z0.shape), z0.shape[-1]))
    if system.shape[-1] != control_gradient.shape[-1]:
        raise ValueError("func did not return a tensor with the same number of input channels as X.derivative "
                         "returned. func returned shape {} (meaning {} channels), whilst X.derivative returned shape "
                         "{} (meaning {} channels)."
                         "".format(tuple(system.shape), system.shape[-1], tuple(control_gradient.shape),
                                   control_gradient.shape[-1]))




def _check_compatability(X, func, z0, t):
    if not hasattr(X, 'derivative'):
        raise ValueError("X must have a 'derivative' method.")
    control_gradient = X.derivative(t[0])
    if hasattr(func, 'prod'):
        raise ValueError('prod is not supported')
    else:
        is_prod = False
        system = func([t[0], z0])

    if isinstance(z0, tf.Tensor):
        is_tensor = True
        if not isinstance(control_gradient, tf.Tensor):
            raise ValueError("z0 is a tensor and so X.derivative must return a tensor as well.")
        if is_prod:
            if not isinstance(vector_field, tf.Tensor):
                raise ValueError("z0 is a tensor and so func.prod must return a tensor as well.")
            _check_compatability_per_tensor_prod(control_gradient, vector_field, z0)
        else:
            if not isinstance(system, tf.Tensor):
                raise ValueError("z0 is a tensor and so func must return a tensor as well.")
            _check_compatability_per_tensor_forward(control_gradient, system, z0)

    elif isinstance(z0, (tuple, list)):
        is_tensor = False
        if not isinstance(control_gradient, (tuple, list)):
            raise ValueError("z0 is a tuple/list and so X.derivative must return a tuple/list as well.")
        if len(z0) != len(control_gradient):
            raise ValueError("z0 and X.derivative(t) must be tuples of the same length.")

        if not isinstance(system, (tuple, list)):
            raise ValueError("z0 is a tuple/list and so func must return a tuple/list as well.")
        if len(z0) != len(system):
            raise ValueError("z0 and func(t, z) must be tuples of the same length.")
        for control_gradient_, system_, z0_ in zip(control_gradient, system, z0):
            if not isinstance(control_gradient_, tf.Tensor):
                raise ValueError("X.derivative must return a tensor or tuple of tensors.")
            if not isinstance(system_, tf.Tensor):
                raise ValueError("func must return a tensor or tuple/list of tensors.")
            _check_compatability_per_tensor_forward(control_gradient_, system_, z0_)

    else:
        raise ValueError("z0 must either a tensor or a tuple/list of tensors.")

    return is_tensor, is_prod


class _VectorField(layers.Layer):
    def __init__(self, X, func, is_tensor, is_prod):
        super(_VectorField, self).__init__()

        self.X = X
        self.func = func
        self.is_tensor = is_tensor
        self.is_prod = is_prod
        
        assert is_prod == False

        # torchsde backend
        self.sde_type = getattr(func, "sde_type", "stratonovich")
        self.noise_type = getattr(func, "noise_type", "additive")

    def f(self, t, z):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.X.derivative(t)

        # vector_field is of shape (..., hidden_channels, input_channels)
        vector_field = self.func([t, z])
        if self.is_tensor:
            # out is of shape (..., hidden_channels)
            # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
            out = tf.squeeze(vector_field @ tf.expand_dims(control_gradient, axis=-1), axis=-1)
        else:
            out = tuple(tf.squeeze(vector_field_ @ tf.expand_dims(control_gradient_, axis =-1), axis = -1)
                        for vector_field_, control_gradient_ in zip(vector_field, control_gradient))

        return out


    def g(self, t, z):
        return tf.expand_dims(tf.zeros(z.shape), axis=-1)


def cdeint(X, func, z0, t, adjoint=True, **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(s, z_s) dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        X: The control. This should be a instance of `torch.nn.Module`, with a `derivative` method. For example
            `torchcde.CubicSpline`. This represents a continuous path derived from the data. The
            derivative at a point will be computed via `X.derivative(t)`, where t is a scalar tensor. The returned
            tensor should have shape (..., input_channels), where '...' is some number of batch dimensions and
            input_channels is the number of channels in the input path.
        func: Should be a callable describing the vector field f(t, z). If using `adjoint=True` (the default), then
            should be an instance of `torch.nn.Module`, to collect the parameters for the adjoint pass. Will be called
            with a scalar tensor t and a tensor z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `X` arguments as above. The '...' corresponds to some number of batch dimensions. If it
            has a method `prod` then that will be called to calculate the matrix-vector product f(t, z) dX_t/dt, via
            `func.prod(t, z, dXdt)`.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate. Defaults to True.
        backend: Either "torchdiffeq" or "torchsde". Which library to use for the solvers. Note that if using torchsde
            that the Brownian motion component is completely ignored -- so it's still reducing the CDE to an ODE --
            but it makes it possible to e.g. use an SDE solver there as the ODE/CDE solver here, if for some reason
            that's desired.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq (the most common are `rtol`, `atol`,
            `method`, `options`) or the sdeint solver of torchsde.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(s, z_s)dX_s, where t_i = t[i].
        This will be a tensor of shape (..., len(t), hidden_channels).

    Raises:
        ValueError for malformed inputs.

    Note:
        Supports tupled input, i.e. z0 can be a tuple of tensors, and X.derivative and func can return tuples of tensors
        of the same length.

    Warnings:
        Note that the returned tensor puts the sequence dimension second-to-last, rather than first like in
        `torchdiffeq.odeint` or `torchsde.sdeint`.
    """

    # Reduce the default values for the tolerances because CDEs are difficult to solve with the default high tolerances.

    is_tensor, is_prod = _check_compatability(X, func, z0, t)

    if adjoint and 'adjoint_params' not in kwargs:
        raise ValueError('adjoint and adjoint_params should be provided')

    vector_field = _VectorField(X=X, func=func, is_tensor=is_tensor, is_prod=is_prod)
    
    
    # backend == "tfsde"
    sdeint = tfsde.sdeint_adjoint if adjoint else tfsde.sdeint
    kwargs['adjoint_params'] = kwargs['adjoint_params'] + tuple(vector_field.trainable_variables)
    out = sdeint(sde=vector_field, y0=z0, ts=t, **kwargs)

    if is_tensor:
        batch_dims = list(range(1, len(out.shape) - 1))
        perm = batch_dims + [0, len(out.shape)-1]
        out = tf.transpose(out, perm)
    else:
        out_ = []
        for outi in out:
            batch_dims = range(1, len(outi.shape) - 1)
            perm = batch_dims + [0, len(outi.shape)-1]
            outi = tf.transpose(outi, perm)
            out_.append(outi)
        out = tuple(out_)

    return out
