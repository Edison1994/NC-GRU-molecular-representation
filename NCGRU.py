# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module implementing RNN Cells.
This module provides a number of basic commonly used RNN cells, such as LSTM
(Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number of
operators that allow adding dropouts, projections, or embeddings for inputs.
Constructing multi-layer cells is supported by the class `MultiRNNCell`, or by
calling the `rnn` ops several times.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

import numpy as np

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

# This can be used with self.assertRaisesRegexp for assert_like_rnncell.
ASSERT_LIKE_RNNCELL_ERROR_REGEXP = "is not an RNNCell"


def assert_like_rnncell(cell_name, cell):
  """Raises a TypeError if cell is not like an RNNCell.
  NOTE: Do not rely on the error message (in particular in tests) which can be
  subject to change to increase readability. Use
  ASSERT_LIKE_RNNCELL_ERROR_REGEXP.
  Args:
    cell_name: A string to give a meaningful error referencing to the name
      of the functionargument.
    cell: The object which should behave like an RNNCell.
  Raises:
    TypeError: A human-friendly exception.
  """
  conditions = [
      hasattr(cell, "output_size"),
      hasattr(cell, "state_size"),
      hasattr(cell, "zero_state"),
      callable(cell),
  ]
  errors = [
      "'output_size' property is missing",
      "'state_size' property is missing",
      "'zero_state' method is missing",
      "is not callable"
  ]

  if not all(conditions):

    errors = [error for error, cond in zip(errors, conditions) if not cond]
    raise TypeError("The argument {!r} ({}) is not an RNNCell: {}.".format(
        cell_name, cell, ", ".join(errors)))


def _concat(prefix, suffix, static=False):
  """Concat that enables int, Tensor, or TensorShape values.
  This function takes a size specification, which can be an integer, a
  TensorShape, or a Tensor, and converts it into a concatenated Tensor
  (if static = False) or a list of integers (if static = True).
  Args:
    prefix: The prefix; usually the batch size (and/or time step size).
      (TensorShape, int, or Tensor.)
    suffix: TensorShape, int, or Tensor.
    static: If `True`, return a python list with possibly unknown dimensions.
      Otherwise return a `Tensor`.
  Returns:
    shape: the concatenation of prefix and suffix.
  Raises:
    ValueError: if `suffix` is not a scalar or vector (or TensorShape).
    ValueError: if prefix or suffix was `None` and asked for dynamic
      Tensors out.
  """
  if isinstance(prefix, ops.Tensor):
    p = prefix
    p_static = tensor_util.constant_value(prefix)
    if p.shape.ndims == 0:
      p = array_ops.expand_dims(p, 0)
    elif p.shape.ndims != 1:
      raise ValueError("prefix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % p)
  else:
    p = tensor_shape.as_shape(prefix)
    p_static = p.as_list() if p.ndims is not None else None
    p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
         if p.is_fully_defined() else None)
  if isinstance(suffix, ops.Tensor):
    s = suffix
    s_static = tensor_util.constant_value(suffix)
    if s.shape.ndims == 0:
      s = array_ops.expand_dims(s, 0)
    elif s.shape.ndims != 1:
      raise ValueError("suffix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % s)
  else:
    s = tensor_shape.as_shape(suffix)
    s_static = s.as_list() if s.ndims is not None else None
    s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
         if s.is_fully_defined() else None)

  if static:
    shape = tensor_shape.as_shape(p_static).concatenate(s_static)
    shape = shape.as_list() if shape.ndims is not None else None
  else:
    if p is None or s is None:
      raise ValueError("Provided a prefix or suffix of None: %s and %s"
                       % (prefix, suffix))
    shape = array_ops.concat((p, s), 0)
  return shape


def _zero_state_tensors(state_size, batch_size, dtype):
  """Create tensors of zeros based on state_size, batch_size, and dtype."""
  def get_state_shape(s):
    """Combine s with batch_size to get a proper tensor shape."""
    c = _concat(batch_size, s)
    size = array_ops.zeros(c, dtype=dtype)
    if not context.executing_eagerly():
      c_static = _concat(batch_size, s, static=True)
      size.set_shape(c_static)
    return size
  return nest.map_structure(get_state_shape, state_size)


@tf_export("nn.rnn_cell.RNNCell")
class RNNCell(base_layer.Layer):
  """Abstract object representing an RNN cell.
  Every `RNNCell` must have the properties below and implement `call` with
  the signature `(output, next_state) = call(input, state)`.  The optional
  third input argument, `scope`, is allowed for backwards compatibility
  purposes; but should be left off for new subclasses.
  This definition of cell differs from the definition used in the literature.
  In the literature, 'cell' refers to an object with a single scalar output.
  This definition refers to a horizontal array of such units.
  An RNN cell, in the most abstract setting, is anything that has
  a state and performs some operation that takes a matrix of inputs.
  This operation results in an output matrix with `self.output_size` columns.
  If `self.state_size` is an integer, this operation also results in a new
  state matrix with `self.state_size` columns.  If `self.state_size` is a
  (possibly nested tuple of) TensorShape object(s), then it should return a
  matching structure of Tensors having shape `[batch_size].concatenate(s)`
  for each `s` in `self.batch_size`.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.
    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size, s] for s in self.state_size`.
      scope: VariableScope for the created subgraph; defaults to class name.
    Returns:
      A pair containing:
      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    if scope is not None:
      with vs.variable_scope(scope,
                             custom_getter=self._rnn_get_variable) as scope:
        return super(RNNCell, self).__call__(inputs, state, scope=scope)
    else:
      scope_attrname = "rnncell_scope"
      scope = getattr(self, scope_attrname, None)
      if scope is None:
        scope = vs.variable_scope(vs.get_variable_scope(),
                                  custom_getter=self._rnn_get_variable)
        setattr(self, scope_attrname, scope)
      with scope:
        return super(RNNCell, self).__call__(inputs, state)

  def _rnn_get_variable(self, getter, *args, **kwargs):
    variable = getter(*args, **kwargs)
    if context.executing_eagerly():
      trainable = variable._trainable  # pylint: disable=protected-access
    else:
      trainable = (
          variable in tf_variables.trainable_variables() or
          (isinstance(variable, tf_variables.PartitionedVariable) and
           list(variable)[0] in tf_variables.trainable_variables()))
    if trainable and variable not in self._trainable_weights:
      self._trainable_weights.append(variable)
    elif not trainable and variable not in self._non_trainable_weights:
      self._non_trainable_weights.append(variable)
    return variable

  @property
  def state_size(self):
    """size(s) of state(s) used by this cell.
    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def build(self, _):
    # This tells the parent Layer object that it's OK to call
    # self.add_variable() inside the call() method.
    pass

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size, state_size]` filled with zeros.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
      the shapes `[batch_size, s]` for each s in `state_size`.
    """
    # Try to use the last cached zero_state. This is done to avoid recreating
    # zeros, especially when eager execution is enabled.
    state_size = self.state_size
    is_eager = context.executing_eagerly()
    if is_eager and hasattr(self, "_last_zero_state"):
      (last_state_size, last_batch_size, last_dtype,
       last_output) = getattr(self, "_last_zero_state")
      if (last_batch_size == batch_size and
          last_dtype == dtype and
          last_state_size == state_size):
        return last_output
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      output = _zero_state_tensors(state_size, batch_size, dtype)
    if is_eager:
      self._last_zero_state = (state_size, batch_size, dtype, output)
    return output


class LayerRNNCell(RNNCell):
  """Subclass of RNNCells that act like proper `tf.Layer` objects.
  For backwards compatibility purposes, most `RNNCell` instances allow their
  `call` methods to instantiate variables via `tf.get_variable`.  The underlying
  variable scope thus keeps track of any variables, and returning cached
  versions.  This is atypical of `tf.layer` objects, which separate this
  part of layer building into a `build` method that is only called once.
  Here we provide a subclass for `RNNCell` objects that act exactly as
  `Layer` objects do.  They must provide a `build` method and their
  `call` methods do not access Variables `tf.get_variable`.
  """

  def __call__(self, inputs, state, scope=None, *args, **kwargs):
    """Run this RNN cell on inputs, starting from the given state.
    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size, s] for s in self.state_size`.
      scope: optional cell scope.
      *args: Additional positional arguments.
      **kwargs: Additional keyword arguments.
    Returns:
      A pair containing:
      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    # Bypass RNNCell's variable capturing semantics for LayerRNNCell.
    # Instead, it is up to subclasses to provide a proper build
    # method.  See the class docstring for more details.
    return base_layer.Layer.__call__(self, inputs, state, scope=scope,
                                     *args, **kwargs)


@tf_export("nn.rnn_cell.BasicRNNCell")
class NCGRU(LayerRNNCell):
  """The most basic RNN cell.
  Args:
    num_units: int, The number of units in the RNN cell. Same as hidden_size in Kyle's code
    activation: Nonlinearity to use.  Default: `modReLU`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
    dtype: Default dtype of the layer (default of `None` means use the type
      of the first input). Required when `build` is called before `call`.
  """

  def __init__(self, num_units, D = None, activation='modReLU', reuse=None, name=None, dtype=None):
    super(NCGRU, self).__init__(_reuse=reuse, name=name, dtype=dtype)
    
    self.gate_kernel_initializer = tf.keras.initializers.he_normal() #init_ops.glorot_uniform_initializer()
    self.gate_bias_initializer = init_ops.zeros_initializer()

    self.UC_kernel_initializer = tf.keras.initializers.he_normal() #init_ops.glorot_uniform_initializer()
    self.bias_C_initializer = init_ops.random_uniform_initializer(-0.1, 0.1)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation

    if D is None:
      self._D = np.identity(self._num_units, dtype=np.float32)
    else:
      self._D = D.astype(np.float32)

    # Initialization of skew-symmetric matrix
    s = np.random.uniform(0, np.pi/2.0, size=int(np.floor(self._num_units/2.0)))

    s = -np.sqrt((1.0 - np.cos(s))/(1.0 + np.cos(s)))

    z = np.zeros(s.size)

    if self._num_units % 2 == 0:
      diag = np.hstack(zip(s, z))[:-1]
    else:
      diag = np.hstack(zip(s,z))

    A_init = np.diag(diag, k=1)

    A_init = A_init - A_init.T

    self.A_init = A_init.astype(np.float32)

    I = np.identity(self._num_units)

    Z_init = np.linalg.lstsq(I + self.A_init, I - self.A_init)[0].astype(np.float32)

    self.W_init = np.matmul(Z_init, self._D)

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units




  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    
    #----------------------------- gates variables -----------------------------
    
    self._gate_kernel = self.add_variable("gate_kernel",
        shape=[input_depth + self._num_units, 2 * self._num_units],
        initializer=self.gate_kernel_initializer)
        
    self._gate_bias = self.add_variable("gate_bias",
        shape=[2 * self._num_units],
        initializer=(self.gate_bias_initializer
                     if self.gate_bias_initializer is not None else
                     init_ops.constant_initializer(1.0, dtype=self.dtype)))
                     
    #--------------------------- sco cell variables ----------------------------

    self._A = self.add_variable("weight_A",
        shape=[self._num_units, self._num_units],
        initializer = init_ops.constant_initializer(self.A_init))

    self._WC = self.add_variable("weight_WC", 
        shape=[self._num_units, self._num_units],
        initializer = init_ops.constant_initializer(self.W_init))

    self._UC = self.add_variable("weight_UC", 
        shape=[input_depth, self._num_units],
        initializer=(self.UC_kernel_initializer
                     if self.UC_kernel_initializer is not None else
                     init_ops.random_uniform_initializer(-0.01, 0.01)))

    self._biasC = self.add_variable("bias_C", shape=[self._num_units],
        initializer=(self.bias_C_initializer
                     if self.bias_C_initializer is not None else
                     init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    """NC-GRU"""
    gate_inputs = math_ops.matmul(tf.concat([inputs, state], 1), self._gate_kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    value = tf.sigmoid(gate_inputs)
    r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    candidate = math_ops.matmul(inputs, self._UC) + math_ops.matmul(r_state, self._WC)
    if self._activation == 'modReLU':
      c = tf.nn.relu(nn_ops.bias_add(tf.abs(candidate), self._biasC))*(tf.sign(candidate))
    else:
      c = nn_ops.bias_add(candidate, self._biasC)
      c = self._activation(c)
      
    new_h = u * state + (1 - u) * c
    
    return new_h, new_h
    