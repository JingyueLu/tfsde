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

import warnings

import tensorflow as tf


def assert_no_grad(names, maybe_tensors):
    for name, maybe_tensor in zip(names, maybe_tensors):
        if hasattr(maybe_tensor, 'trainable'):
            if maybe_tensor.trainable != False:
                raise ValueError(f"Argument {name} must not require gradient.")


def handle_unused_kwargs(unused_kwargs, msg=None):
    if len(unused_kwargs) > 0:
        if msg is not None:
            warnings.warn(f"{msg}: Unexpected arguments {unused_kwargs}")
        else:
            warnings.warn(f"Unexpected arguments {unused_kwargs}")


def flatten(sequence):
    return tf.concat([tf.reshape(p, -1) for p in sequence], axis = 0) if len(sequence) > 0 else tf.constant([])



def is_strictly_increasing(ts):
    return all(x < y for x, y in zip(ts[:-1], ts[1:]))



def seq_add(*seqs):
    return [sum(seq) for seq in zip(*seqs)]



def batch_mvp(m, v):
    return tf.squeeze(tf.matmul(m, tf.expand_dims(v, axis=-1)), axis=-1)




def flat_to_shape(flat_tensor, shapes):
    """Convert a flat tensor to a list of tensors with specified shapes.

    `flat_tensor` must have exactly the number of elements as stated in `shapes`.
    """
    numels = [tf.math.reduce_prod(shape) for shape in shapes]
    return [tf.reshape(flat, shape) for flat, shape in zip(tf.split(flat_tensor,numels), shapes)]
