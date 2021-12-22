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

from tensorflow.keras import layers
from . import misc
from ..settings import NOISE_TYPES, SDE_TYPES
from ..types import Tensor


class BaseSDE(abc.ABC, layers.Layer):
    """Base class for all SDEs.

    Inheriting from this class ensures `noise_type` and `sde_type` are valid attributes, which the solver depends on.
    """

    def __init__(self, noise_type, sde_type):
        super(BaseSDE, self).__init__()
        if noise_type not in NOISE_TYPES:
            raise ValueError(f"Expected noise type in {NOISE_TYPES}, but found {noise_type}")
        if sde_type not in SDE_TYPES:
            raise ValueError(f"Expected sde type in {SDE_TYPES}, but found {sde_type}")
        # Making these Python properties breaks `torch.jit.script`.
        self.noise_type = noise_type
        self.sde_type = sde_type


class ForwardSDE(BaseSDE):

    def __init__(self, sde):
        super(ForwardSDE, self).__init__(sde_type=sde.sde_type, noise_type=sde.noise_type)
        self._base_sde = sde

        # Register the core functions. This avoids polluting the codebase with if-statements and achieves speed-ups
        # by making sure it's a one-time cost.


        self.f = getattr(sde, 'f', self.f_default)
        self.g = getattr(sde, 'g', self.g_default)
        self.f_and_g = getattr(sde, 'f_and_g', self.f_and_g_default)
        self.g_prod = getattr(sde, 'g_prod', self.g_prod_default)
        self.prod = {
            NOISE_TYPES.diagonal: self.prod_diagonal
        }.get(sde.noise_type, self.prod_default)

    ########################################
    #                  f                   #
    ########################################
    def f_default(self, t, y):
        raise RuntimeError("Method `f` has not been provided, but is required for this method.")

    ########################################
    #                  g                   #
    ########################################
    def g_default(self, t, y):
        raise RuntimeError("Method `g` has not been provided, but is required for this method.")

    ########################################
    #               f_and_g                #
    ########################################

    def f_and_g_default(self, t, y):
        return self.f(t, y), self.g(t, y)

    ########################################
    #                prod                  #
    ########################################

    def prod_diagonal(self, g, v):
        return g * v

    def prod_default(self, g, v):
        return misc.batch_mvp(g, v)

    ########################################
    #                g_prod                #
    ########################################

    def g_prod_default(self, t, y, v):
        return self.prod(self.g(t, y), v)


    def _return_zero(self, t, y, v):  # noqa
        return 0.




class SDEIto(BaseSDE):

    def __init__(self, noise_type):
        super(SDEIto, self).__init__(noise_type=noise_type, sde_type=SDE_TYPES.ito)


class SDEStratonovich(BaseSDE):

    def __init__(self, noise_type):
        super(SDEStratonovich, self).__init__(noise_type=noise_type, sde_type=SDE_TYPES.stratonovich)


