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

"""Optional fused Pallas kernels from the `colabfold-kernels` package.

ColabFold's maintained kernels (https://github.com/mirditalab/colabfold-kernels, MIT)
for attention, LayerNorm and triangle multiplication's gated projection. They are off
unless `global_config.use_pallas` is set, and `colabfold_kernels` is imported only then.
Each selector returns a callable, or None for the stock XLA code.

Only the bf16 Pallas kernels are used: they need an NVIDIA GPU of compute capability
8.0 or newer, which is what the caller is expected to check before setting the flag.
The package's fp16 and CUDA "legacy" kernels for older GPUs are never selected here.

The kernels are written for jax >= 0.6.2. `_install_pallas_compat` adds the few names
they use that jax 0.5.3 lacks, so they also run on AlphaFold's pinned jax.
"""

import functools

import jax.numpy as jnp

# Additive mask value handed to the kernels; they treat anything below -1e3 as masked.
_MASK_BIAS = -1e9


def _install_pallas_compat():
  """Add jax >= 0.6 Pallas names to `pallas.triton` where they are missing.

  jax 0.6 renamed `TritonCompilerParams` to `CompilerParams` and gave
  `pallas.triton` its own `load`, `store`, `atomic_add` and `dot`. On jax 0.5.3 the
  same operations live in `jax.experimental.pallas`, with an explicit index argument
  (None indexes the whole ref or view). On newer jax nothing is changed.
  """
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import triton as plgpu

  def load(ref, *, mask=None, other=None, **kwargs):
    return pl.load(ref, None, mask=mask, other=other, **kwargs)

  def store(ref, val, *, mask=None, **kwargs):
    return pl.store(ref, None, val, mask=mask, **kwargs)

  for name, value in (
      ('CompilerParams', getattr(plgpu, 'TritonCompilerParams', None)),
      ('load', load),
      ('store', store),
      ('atomic_add', getattr(pl, 'atomic_add', None)),
      ('dot', getattr(pl, 'dot', None)),
  ):
    if not hasattr(plgpu, name) and value is not None:
      setattr(plgpu, name, value)


@functools.lru_cache(maxsize=None)
def fused_ops():
  """The `colabfold_kernels.fused_ops` module, imported on first use."""
  _install_pallas_compat()
  from colabfold_kernels import fused_ops as ops  # pylint: disable=g-import-not-at-top
  return ops


def enabled(global_config) -> bool:
  return bool(global_config.get('use_pallas', False))


def _selector_config(global_config):
  # Only the Pallas backend: never the package's fp16 CUDA kernels for sm < 80.
  return {
      'use_pallas': True,
      'kernel_backend': 'pallas',
      'compute_capability': global_config.get('compute_capability', None),
  }


def attention(global_config, dtype, key_dim: int, value_dim: int):
  """f(q, k, v, mask_bias, nonbatched_bias, scale) on [b, h, S, c], or None."""
  if not enabled(global_config) or jnp.dtype(dtype) != jnp.dtype(jnp.bfloat16):
    return None
  return fused_ops().attention(
      _selector_config(global_config), dtype, key_dim, value_dim)


def attention_fused(global_config, config, act) -> bool:
  """True if Attention(config) on `act` runs the fused kernel, which needs no subbatching."""
  num_head = config.num_head
  key_dim = config.get('key_dim', int(act.shape[-1])) // num_head
  value_dim = config.get('value_dim', int(act.shape[-1])) // num_head
  return attention(global_config, act.dtype, key_dim, value_dim) is not None


def mask_to_bias(mask, dtype):
  """AlphaFold's 0/1 attention mask as the kernels' additive key bias, or None.

  The kernels mask keys only: they read a bias of shape [batch, 1, 1, N_keys]. Any
  mask that varies over heads or queries is declined (None), and the caller then
  uses the stock XLA attention.
  """
  if mask.ndim != 4 or mask.shape[1] != 1 or mask.shape[2] != 1:
    return None
  return jnp.where(mask, 0.0, _MASK_BIAS).astype(dtype)


def layer_norm(global_config, dtype):
  """f(x, scale, offset, *, eps) normalising the last axis, or None."""
  if not enabled(global_config) or jnp.dtype(dtype) != jnp.dtype(jnp.bfloat16):
    return None
  return fused_ops().layer_norm(_selector_config(global_config), dtype)


def gated_dual_proj(global_config, dtype):
  """Triangle multiplication's fused projection + gate + mask, or None."""
  if not enabled(global_config) or jnp.dtype(dtype) != jnp.dtype(jnp.bfloat16):
    return None
  return fused_ops().gated_dual_proj(_selector_config(global_config), dtype)


# Transitions are not quadratic, so with the fused kernels they are chunked to a
# ~256 MiB intermediate rather than `subbatch_size` rows (ColabFold's measured choice).
_TRANSITION_BUDGET_BYTES = 256 * 1024 * 1024


def transition_subbatch(global_config, shape, num_intermediate: int, dtype):
  """Subbatch size for a Transition: the configured one unless fused kernels are on."""
  configured = global_config.subbatch_size
  if configured is None or not enabled(global_config):
    return configured
  per_row = num_intermediate * jnp.dtype(dtype).itemsize
  for dim in shape[1:-1]:
    per_row *= int(dim)
  if per_row <= 0:
    return configured
  return max(configured, min(shape[0], _TRANSITION_BUDGET_BYTES // per_row))
