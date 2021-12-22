# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings


from . import base_sde
from . import methods
from . import misc
from . import sdeint
from .adjoint_sde import AdjointSDE
from .._brownian import BaseBrownian, ReverseBrownian
from ..settings import METHODS, NOISE_TYPES, SDE_TYPES
from ..types import Any, Dict, Optional, Scalar, Tensor, Tensors, TensorOrTensors, Vector

class _SdeintAdjointMethod:

    def __init__(self, sde, ts, dt, bm, solver, method, adjoint_method, 
                dt_min, adjoint_options, len_extras, *extras_and_adjoint_params):
        
        # Initialize (In tf, the input for a function with customised gradients has to be 
        # a list of tensors)
        self.sde = sde; self.ts = ts; self.dt = dt; self.bm = bm
        self.solver = solver; self.method = method; self.adjoint_method = adjoint_method
        self.dt_min = dt_min 
        self.adjoint_options = adjoint_options;
        self.len_extras = len_extras
        self.extras_and_adjoint_params = extras_and_adjoint_params
        self.extra_solver_state = self.extras_and_adjoint_params[:self.len_extras]
        self.adjoint_params = self.extras_and_adjoint_params[self.len_extras:]
    

        self.f = tf.custom_gradient(lambda x: _SdeintAdjointMethod._f(self, x))


    def _f(self, y0):
        
        ys, extra_solver_state = self.solver.integrate(y0, self.ts, self.extra_solver_state)

        self.extra_solver_state = extra_solver_state

        
        def grad_fn(upstream, variables):  # noqa
            extra_solver_state = self.extra_solver_state
            adjoint_params = self.adjoint_params

            aug_state = [ys[-1], upstream[-1]] + [tf.zeros(param.shape) for param in adjoint_params]
            shapes = [t.shape for t in aug_state]
            aug_state = misc.flatten(aug_state)
            aug_state = tf.expand_dims(aug_state, axis=0)  # dummy batch dimension
            adjoint_sde = AdjointSDE(self.sde, adjoint_params, shapes)
            reverse_bm = ReverseBrownian(self.bm)

            solver_fn = methods.select(method=self.adjoint_method, sde_type=adjoint_sde.sde_type)
            solver = solver_fn(
                sde=adjoint_sde,
                bm=reverse_bm,
                dt=self.dt,
                dt_min=self.dt_min,
                options=self.adjoint_options
            )
            
            internal_adjoint = _SdeintAdjointMethod(adjoint_sde,
                                                    None,
                                                    self.dt,
                                                    reverse_bm,
                                                    solver,
                                                    self.adjoint_method,
                                                    self.adjoint_method,
                                                    self.dt_min,
                                                    self.adjoint_options,
                                                    len(extra_solver_state),
                                                    *extra_solver_state,
                                                    *adjoint_params)



            for i in range(ys.shape[0] - 1, 0, -1):
                internal_adjoint.ts = tf.stack([-self.ts[i], -self.ts[i - 1]])
                aug_state = tf.stop_gradient(internal_adjoint._f(aug_state))

                aug_state = misc.flat_to_shape(aug_state.squeeze(0), shapes)
                aug_state[0] = ys[i - 1]
                aug_state[1] = aug_state[1] + grad_ys[i - 1]
                if i != 1:
                    aug_state = misc.flatten(aug_state)
                    aug_state = aug_state.unsqueeze(0)  # dummy batch dimension

            if saved_extras_for_backward:
                out = aug_state[1:]
            else:
                out = [aug_state[1]] + ([None] * len_extras) + aug_state[2:]

            return upstream

        return ys, grad_fn


def sdeint_adjoint(sde: layers.Layer,
                   y0: Tensor,
                   ts: Vector,
                   bm: Optional[BaseBrownian] = None,
                   method: Optional[str] = None,
                   adjoint_method: Optional[str] = None,
                   dt: Scalar = 1e-3,
                   dt_min: Scalar = 1e-5,
                   options: Optional[Dict[str, Any]] = None,
                   adjoint_options: Optional[Dict[str, Any]] = None,
                   adjoint_params=None,
                   names: Optional[Dict[str, str]] = None,
                   extra_solver_state: Optional[Tensors] = None,
                   **unused_kwargs) -> TensorOrTensors:
    """Numerically integrate an SDE with stochastic adjoint support.

    Args:
        sde (keras.Model): Object with methods `f` and `g` representing the
            drift and diffusion. The output of `g` should be a single tensor of
            size (batch_size, d) for diagonal noise SDEs or (batch_size, d, m)
            for SDEs of other noise types; d is the dimensionality of state and
            m is the dimensionality of Brownian motion.
        y0 (Tensor): A tensor for the initial state.
        ts (Tensor or sequence of float): Query times in non-descending order.
            The state at the first time of `ts` should be `y0`.
        bm (Brownian, optional): A 'BrownianInterval', `BrownianPath` or
            `BrownianTree` object. Should return tensors of size (batch_size, m)
            for `__call__`. Defaults to `BrownianInterval`.
        method (str, optional): Numerical integration method to use. Must be
            compatible with the SDE type (Ito/Stratonovich) and the noise type
            (scalar/additive/diagonal/general). Defaults to a sensible choice
            depending on the SDE type and noise type of the supplied SDE.
        adjoint_method (str, optional): Name of numerical integration method for
            backward adjoint solve. Defaults to a sensible choice depending on
            the SDE type and noise type of the supplied SDE.
        dt (float, optional): The constant step size or initial step size for
            adaptive time-stepping.
        adaptive (bool, optional): If `True`, use adaptive time-stepping.
        adjoint_adaptive (bool, optional): If `True`, use adaptive time-stepping
            for the backward adjoint solve.
        rtol (float, optional): Relative tolerance.
        adjoint_rtol (float, optional): Relative tolerance for backward adjoint
            solve.
        atol (float, optional): Absolute tolerance.
        adjoint_atol (float, optional): Absolute tolerance for backward adjoint
            solve.
        dt_min (float, optional): Minimum step size during integration.
        options (dict, optional): Dict of options for the integration method.
        adjoint_options (dict, optional): Dict of options for the integration
            method of the backward adjoint solve.
        adjoint_params (Sequence of Tensors, optional): Tensors whose gradient
            should be obtained with the adjoint. If not specified, defaults to
            the parameters of `sde`.
        names (dict, optional): Dict of method names for drift and diffusion.
            Expected keys are "drift" and "diffusion". Serves so that users can
            use methods with names not in `("f", "g")`, e.g. to use the
            method "foo" for the drift, we supply `names={"drift": "foo"}`.
        logqp (bool, optional): If `True`, also return the log-ratio penalty.
        extra (bool, optional): If `True`, also return the extra hidden state
            used internally in the solver.
        extra_solver_state: (tuple of Tensors, optional): Additional state to
            initialise the solver with. Some solvers keep track of additional
            state besides y0, and this offers a way to optionally initialise
            that state.

    Returns:
        A single state tensor of size (T, batch_size, d).
        if logqp is True, then the log-ratio penalty is also returned.
        If extra is True, the any extra internal state of the solver is also
        returned.

    Raises:
        ValueError: An error occurred due to unrecognized noise type/method,
            or `sde` is missing required methods.

    Note:
        The backward pass is much more efficient with Stratonovich SDEs than
        with Ito SDEs.

    Note:
        Double-backward is supported for Stratonovich SDEs. Doing so will use
        the adjoint method to compute the gradient of the adjoint. (i.e. rather
        than backpropagating through the numerical solver used for the
        adjoint.) The same `adjoint_method`, `adjoint_adaptive`, `adjoint_rtol,
        `adjoint_atol`, `adjoint_options` will be used for the second-order
        adjoint as is used for the first-order adjoint.
    """
    misc.handle_unused_kwargs(unused_kwargs, msg="`sdeint_adjoint`")
    #import pdb; pdb.set_trace()
    del unused_kwargs

    if adjoint_params is None and not isinstance(sde, layers.Layer):
        raise ValueError('`sde` must be an instance of keras.layers.Layer to specify the adjoint parameters; alternatively they '
                         'can be specified explicitly via the `adjoint_params` argument. If there are no parameters '
                         'then it is allowable to set `adjoint_params=()`.')

    sde, y0, ts, bm, method, options = sdeint.check_contract(sde, y0, ts, bm, method, options, names)
    misc.assert_no_grad(['ts', 'dt','dt_min'],
                        [ts, dt, dt_min])
    
    # trainable_weights and trainable_variables are the same
    adjoint_params = tuple(sde.trainable_variables) if adjoint_params is None else tuple(adjoint_params)
    adjoint_params = filter(lambda x: hasattr(x, 'trainable'), adjoint_params)
    adjoint_params = filter(lambda x: x.trainable, adjoint_params)
    adjoint_method = _select_default_adjoint_method(sde, method, adjoint_method)
    adjoint_options = {} if adjoint_options is None else adjoint_options.copy()

    # Note that all of these warnings are only applicable for reversible solvers with sdeint_adjoint; none of them
    # apply to sdeint.
    if method == METHODS.reversible_heun:
        if adjoint_method != METHODS.adjoint_reversible_heun:
            warnings.warn(f"method={repr(method)}, but adjoint_method!={repr(METHODS.adjoint_reversible_heun)}.")

        num_steps = (ts - ts[0]) / dt
        if not tf.experimental.numpy.allclose(num_steps, tf.round(num_steps)):
            # NOTE need to check again and understand it
            warnings.warn(f"The spacing between time points `ts` is not an integer multiple of the time step `dt`. "
                          f"This means that the backward pass (which is forced to step to each of `ts` to get "
                          f"dL/dy(t) for t in ts) will not perfectly mimick the forward pass (which does not step "
                          f"to each `ts`, and instead interpolates to them). This means that "
                          f"method={repr(method)} may not be perfectly accurate.")

    solver_fn = methods.select(method=method, sde_type=sde.sde_type)
    solver = solver_fn(
        sde=sde,
        bm=bm,
        dt=dt,
        dt_min=dt_min,
        options=options
    )
    if extra_solver_state is None:
        # extra_solver_state: a tuple of (drift, diffusion, y0)
        extra_solver_state = solver.init_extra_solver_state(ts[0], y0)

    SdeAdjoint = _SdeintAdjointMethod(
        sde, ts, dt, bm, solver, method, adjoint_method,  dt_min,
        adjoint_options, len(extra_solver_state), *extra_solver_state, *adjoint_params
    )
    
    ys = SdeAdjoint.f(y0)

    return ys


def _select_default_adjoint_method(sde: base_sde.ForwardSDE, method: str, adjoint_method: Optional[str]) -> str:
    """Select the default method for adjoint computation based on the noise type of the forward SDE."""
    if adjoint_method is not None:
        return adjoint_method
    elif method == METHODS.reversible_heun:
        return METHODS.adjoint_reversible_heun
    else:
        raise ValueError('This implementation only supports Reversible_heun method and its adjoint method.')
