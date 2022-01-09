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

from . import base_sde
from . import misc
from ..settings import NOISE_TYPES, SDE_TYPES
from ..types import Sequence, TensorOrTensors


class AdjointSDE(base_sde.BaseSDE):

    def __init__(self,
                 forward_sde: base_sde.ForwardSDE,
                 params: TensorOrTensors,
                 shapes: Sequence[tf.shape]):
        # There's a mapping from the noise type of the forward SDE to the noise type of the adjoint.
        # Usually, these two aren't the same, e.g. when the forward SDE has additive noise, the adjoint SDE's diffusion
        # is a linear function of the adjoint variable, so it is not of additive noise.
        sde_type = forward_sde.sde_type
        noise_type = {
            NOISE_TYPES.general: NOISE_TYPES.general,
            NOISE_TYPES.additive: NOISE_TYPES.general,
            NOISE_TYPES.scalar: NOISE_TYPES.scalar,
            NOISE_TYPES.diagonal: NOISE_TYPES.diagonal,
        }.get(forward_sde.noise_type)
        super(AdjointSDE, self).__init__(sde_type=sde_type, noise_type=noise_type)

        self.forward_sde = forward_sde
        self.params = params
        self._shapes = shapes


    ########################################
    #            Helper functions          #
    ########################################

    def get_state(self, t, y_aug, v=None, extra_states=False):
        """Unpacks y_aug, whilst enforcing the necessary checks so that we can calculate derivatives wrt state."""

        # These leaf checks are very important.
        # get_state is used where we want to compute:
        # ```
        # with torch.enable_grad():
        #     s = some_function(y)
        #     torch.autograd.grad(s, [y] + params, ...)
        # ```
        # where `some_function` implicitly depends on `params`.
        # However if y has history of its own then in principle it could _also_ depend upon `params`, and this call to
        # `grad` will go all the way back to that. To avoid this, we require that every input tensor be a leaf tensor.
        #
        # This is also the reason for the `y0.detach()` in adjoint.py::_SdeintAdjointMethod.forward. If we don't detach,
        # then y0 may have a history and these checks will fail. This is a spurious failure as
        # `torch.autograd.Function.forward` has an implicit `torch.no_grad()` guard, i.e. we definitely don't want to
        # use its history there.

        if extra_states:
            shapes = self._shapes
        else:
            shapes = self._shapes[:2]
        numel = sum(tf.math.reduce_prod(shape) for shape in shapes)
        y, adj_y, *extra_states = misc.flat_to_shape(tf.squeeze(y_aug, 0)[:numel], shapes)

        return y, adj_y, extra_states

