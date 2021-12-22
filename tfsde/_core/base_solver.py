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

import abc
import warnings

import tensorflow as tf

from . import better_abc
from . import interp
from .base_sde import BaseSDE
from .._brownian import BaseBrownian
from ..settings import NOISE_TYPES
from ..types import Scalar, Tensor, Dict, Tensors, Tuple


class BaseSDESolver(metaclass=better_abc.ABCMeta):
    """API for solvers with possibly adaptive time stepping."""

    strong_order = better_abc.abstract_attribute()
    weak_order = better_abc.abstract_attribute()
    sde_type = better_abc.abstract_attribute()
    noise_types = better_abc.abstract_attribute()
    levy_area_approximations = better_abc.abstract_attribute()

    def __init__(self,
                 sde: BaseSDE,
                 bm: BaseBrownian,
                 dt: Scalar,
                 dt_min: Scalar,
                 options: Dict,
                 **kwargs):
        super(BaseSDESolver, self).__init__(**kwargs)
        if sde.sde_type != self.sde_type:
            raise ValueError(f"SDE is of type {sde.sde_type} but solver is for type {self.sde_type}")
        if sde.noise_type not in self.noise_types:
            raise ValueError(f"SDE has noise type {sde.noise_type} but solver only supports noise types "
                             f"{self.noise_types}")
        if bm.levy_area_approximation not in self.levy_area_approximations:
            raise ValueError(f"SDE solver requires one of {self.levy_area_approximations} set as the "
                             f"`levy_area_approximation` on the Brownian motion.")
        if sde.noise_type == NOISE_TYPES.scalar and tf.size(bm.shape[1:]) != 1:  # noqa
            raise ValueError("The Brownian motion for scalar SDEs must of dimension 1.")

        self.sde = sde
        self.bm = bm
        self.dt = dt
        self.dt_min = dt_min
        self.options = options

    def __repr__(self):
        return f"{self.__class__.__name__} of strong order: {self.strong_order}, and weak order: {self.weak_order}"

    def init_extra_solver_state(self, t0, y0) -> Tensors:
        return ()

    @abc.abstractmethod
    def step(self, t0: Scalar, t1: Scalar, y0: Tensor, extra0: Tensors) -> Tuple[Tensor, Tensors]:
        """Propose a step with step size from time t to time next_t, with
         current state y.

        Args:
            t0: float or Tensor of size (,).
            t1: float or Tensor of size (,).
            y0: Tensor of size (batch_size, d).
            extra0: Any extra state for the solver.

        Returns:
            y1, where y1 is a Tensor of size (batch_size, d).
            extra1: Modified extra state for the solver.
        """
        raise NotImplementedError

    def integrate(self, y0: Tensor, ts: Tensor, extra0: Tensors) -> Tuple[Tensor, Tensors]:
        """Integrate along trajectory.

        Args:
            y0: Tensor of size (batch_size, d)
            ts: Tensor of size (T,).
            extra0: Any extra state for the solver.

        Returns:
            ys, where ys is a Tensor of size (T, batch_size, d).
            extra_solver_state, which is a tuple of Tensors of shape (T, ...), where ... is arbitrary and
                solver-dependent.
        """
        step_size = self.dt

        prev_t = curr_t = ts[0]
        prev_y = curr_y = y0
        curr_extra = extra0

        ys = [y0]
        prev_error_ratio = None

        for out_t in ts[1:]:
            while curr_t < out_t:
                next_t = min(curr_t + step_size, ts[-1])
                prev_t, prev_y = curr_t, curr_y
                curr_y, curr_extra = self.step(curr_t, next_t, curr_y, curr_extra)
                curr_t = next_t
            ys.append(interp.linear_interp(t0=prev_t, y0=prev_y, t1=curr_t, y1=curr_y, t=out_t))

        return tf.stack(ys, axis=0), curr_extra
