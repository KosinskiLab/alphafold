# Copyright 2026 KosinskiLab
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

"""Tests for the fused-kernel hooks, on CPU.

The real kernels need a GPU. Here `fused_kernels.fused_ops` is replaced by plain-JAX
reference implementations with the same calling conventions, so these tests check
the wiring: the mask-to-bias adapter, layouts, the channel-major split of triangle
multiplication, parameter names, and that the stock path is untouched when off.
"""

import sys
import types
from unittest import mock

from absl.testing import absltest
from alphafold.model import common_modules
from alphafold.model import config
from alphafold.model import fused_kernels
from alphafold.model import modules
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np


def _reference_attention(q, k, v, mask_bias, nonbatched_bias, scale):
  """colabfold_kernels' attention contract, [b, h, S, c] layout, in plain JAX."""
  logits = jnp.einsum('bhqc,bhkc->bhqk', q, k).astype(jnp.float32) * scale
  if nonbatched_bias is not None:
    logits += nonbatched_bias[None].astype(jnp.float32)
  logits = jnp.where(mask_bias > -1e3, logits, -1e9)
  weights = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
  return jnp.einsum('bhqk,bhkc->bhqc', weights, v)


def _reference_layer_norm(x, scale, offset, *, eps):
  xf = x.astype(jnp.float32)
  mean = jnp.mean(xf, -1, keepdims=True)
  var = jnp.mean((xf - mean) ** 2, -1, keepdims=True)
  return ((xf - mean) * jax.lax.rsqrt(var + eps) * scale + offset).astype(x.dtype)


def _reference_gated_dual_proj(x, wp, bp, wg, bg, mask, *, split=False,
                               channel_major=False):
  out = mask[:, None] * (x @ wp + bp) * jax.nn.sigmoid(x @ wg + bg)
  if not split:
    return out
  half = out.shape[-1] // 2
  left, right = out[:, :half], out[:, half:]
  if channel_major:
    left, right = left.T, right.T
  return left, right


_FAKE_OPS = types.SimpleNamespace(
    attention=lambda gc, dtype, key_dim, value_dim: _reference_attention,
    layer_norm=lambda gc, dtype: _reference_layer_norm,
    gated_dual_proj=lambda gc, dtype: _reference_gated_dual_proj,
)


def _global_config(use_pallas):
  gc = config.model_config('model_1_multimer_v3').model.global_config
  gc = ml_collections.ConfigDict(gc.to_dict())
  gc.use_pallas = use_pallas
  gc.compute_capability = 90 if use_pallas else None
  return gc


def _run(fn, *args):
  """Init once, then apply with the same params with kernels off and on."""
  transformed = hk.transform(fn)
  params = transformed.init(jax.random.PRNGKey(0), False, *args)
  off = transformed.apply(params, None, False, *args)
  on = transformed.apply(params, None, True, *args)
  return np.asarray(off, np.float32), np.asarray(on, np.float32)


class FusedKernelsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.object(fused_kernels, 'fused_ops', lambda: _FAKE_OPS)
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_off_does_not_import_the_package(self):
    with mock.patch.dict(sys.modules, {'colabfold_kernels': None}):
      gc = _global_config(False)
      self.assertIsNone(fused_kernels.attention(gc, jnp.bfloat16, 32, 32))
      self.assertIsNone(fused_kernels.layer_norm(gc, jnp.bfloat16))
      self.assertIsNone(fused_kernels.gated_dual_proj(gc, jnp.bfloat16))

  def test_only_bf16_is_fused(self):
    gc = _global_config(True)
    self.assertIsNone(fused_kernels.attention(gc, jnp.float32, 32, 32))
    self.assertIsNone(fused_kernels.layer_norm(gc, jnp.float32))
    self.assertIsNotNone(fused_kernels.attention(gc, jnp.bfloat16, 32, 32))

  def test_mask_to_bias(self):
    mask = jnp.array([1.0, 1.0, 0.0])[None, None, None, :]
    bias = fused_kernels.mask_to_bias(mask, jnp.bfloat16)
    self.assertEqual(bias.shape, (1, 1, 1, 3))
    np.testing.assert_array_equal(np.asarray(bias[0, 0, 0, :2], np.float32), 0.0)
    self.assertLess(float(bias[0, 0, 0, 2]), -1e3)
    # A mask that varies over queries is declined: the kernels mask keys only.
    self.assertIsNone(fused_kernels.mask_to_bias(jnp.ones((2, 1, 4, 4)), jnp.bfloat16))

  def test_transition_subbatch(self):
    off, on = _global_config(False), _global_config(True)
    shape = (384, 384, 128)
    self.assertEqual(
        fused_kernels.transition_subbatch(off, shape, 512, jnp.bfloat16), 4)
    wide = fused_kernels.transition_subbatch(on, shape, 512, jnp.bfloat16)
    self.assertGreater(wide, 4)
    self.assertLessEqual(wide, shape[0])

  def test_attention_matches_stock_with_padded_keys(self):
    attention_config = ml_collections.ConfigDict(
        {'num_head': 4, 'gating': True, 'key_dim': 64, 'value_dim': 64})
    b, n, c = 3, 12, 32
    rng = np.random.default_rng(0)
    q_data = jnp.asarray(rng.normal(size=(b, n, c)), jnp.bfloat16)
    key_mask = (np.arange(n) < n - 3).astype(np.float32)
    mask = jnp.asarray(np.broadcast_to(key_mask, (b, n)))[:, None, None, :]
    pair_bias = jnp.asarray(rng.normal(size=(4, n, n)), jnp.bfloat16)

    def fn(use_pallas, q_data, mask, pair_bias):
      return modules.Attention(attention_config, _global_config(use_pallas), c)(
          q_data, q_data, mask, pair_bias)

    off, on = _run(fn, q_data, mask, pair_bias)
    np.testing.assert_allclose(on, off, atol=5e-2, rtol=5e-2)

  def test_triangle_multiplication_matches_stock(self):
    tri_config = config.model_config(
        'model_1_multimer_v3').model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing
    self.assertTrue(tri_config.fuse_projection_weights)
    n, cz = 10, 16
    rng = np.random.default_rng(1)
    pair_act = jnp.asarray(rng.normal(size=(n, n, cz)), jnp.bfloat16)
    pair_mask = jnp.asarray((rng.random((n, n)) > 0.2).astype(np.float32))

    def fn(use_pallas, pair_act, pair_mask):
      gc = _global_config(use_pallas)
      common_modules.set_kernel_context(gc)
      return modules.TriangleMultiplication(tri_config, gc)(pair_act, pair_mask)

    off, on = _run(fn, pair_act, pair_mask)
    self.assertEqual(off.shape, (n, n, cz))
    np.testing.assert_allclose(on, off, atol=5e-2, rtol=5e-2)

  def test_triangle_multiplication_uses_the_stock_parameters(self):
    tri_config = config.model_config(
        'model_1_multimer_v3').model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming
    pair_act = jnp.ones((6, 6, 16), jnp.bfloat16)
    pair_mask = jnp.ones((6, 6), jnp.float32)

    def init(use_pallas):
      gc = _global_config(use_pallas)
      common_modules.set_kernel_context(gc)
      return hk.transform(
          lambda a, m: modules.TriangleMultiplication(tri_config, gc)(a, m)
      ).init(jax.random.PRNGKey(0), pair_act, pair_mask)

    stock, fused = init(False), init(True)
    self.assertEqual(
        jax.tree_util.tree_map(np.shape, stock),
        jax.tree_util.tree_map(np.shape, fused))

  def test_layer_norm_matches_stock(self):
    x = jnp.asarray(np.random.default_rng(2).normal(size=(5, 7, 32)), jnp.bfloat16)

    def fn(use_pallas, x):
      common_modules.set_kernel_context(_global_config(use_pallas))
      return common_modules.LayerNorm(
          axis=[-1], create_scale=True, create_offset=True, name='ln')(x)

    off, on = _run(fn, x)
    np.testing.assert_allclose(on, off, atol=2e-2, rtol=2e-2)
    common_modules.set_kernel_context(_global_config(False))


if __name__ == '__main__':
  absltest.main()
