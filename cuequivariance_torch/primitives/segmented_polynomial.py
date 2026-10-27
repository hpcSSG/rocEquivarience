# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by mlx in 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Dict, List, Optional, OrderedDict, Tuple, Any

import torch
import torch.nn as nn
from cuequivariance_torch.primitives.segmented_polynomial_fused_tp import (
    SegmentedPolynomialFusedTP,
)
from cuequivariance_torch.primitives.segmented_polynomial_indexed_linear import (
    SegmentedPolynomialIndexedLinear,
)
from cuequivariance_torch.primitives.segmented_polynomial_naive import (
    SegmentedPolynomialNaive,
)
from cuequivariance_torch.primitives.segmented_polynomial_uniform_1d import (
    SegmentedPolynomialFromUniform1dJit,
)

import cuequivariance as cue

try:
    import cuequivariance_ops_torch  # noqa: F401

    HAS_CUE_OPS = True
except ImportError:
    HAS_CUE_OPS = False

import time
#from mace.tools.scatter import scatter_sum

#from fasteq.ops.so3_linear import fast_equi_linear
from fasteq.ops.fctp import fast_fctp
from fasteq.ops.uniform1d_jit import fast_uniform1d_jit
from fasteq.ops.stc_uniform1d_jit import fast_stc_uniform1d_jit


import math
from torch.nn.utils.rnn import pad_sequence


STC_PAD_VALUE = -1

def flatten_stp(d: cue.SegmentedTensorProduct) -> cue.SegmentedTensorProduct:
    

    d = d.move_operand(0, -2)
    d = d.flatten_coefficient_modes(force=True)
    d = d.flatten_modes(
        [
            m
            for m in d.subscripts.modes()
            if not all(m in ss for ss in d.subscripts.operands)
        ]
    )
    d = d.consolidate_modes()
    if d.subscripts.modes() == []:
        d = d.append_modes_to_all_operands("u", dict(u=1))
    '''
    for oid in range(0, d.num_operands - 2):
        print(f"oid:{oid}, len d.operands[oid].num_segments:{d.operands[oid].num_segments}")
    '''

    # ops.SymmetricTensorContraction will "symmetrize" for the derivatives so we can sort for the forward pass
    d = d.sort_indices_for_identical_operands(range(0, d.num_operands - 2))

    if len(d.subscripts.modes()) != 1:
        raise NotImplementedError("Different modes are not supported.")

    m = d.subscripts.modes()[0]

    if not all(ss == m for ss in d.subscripts.operands):
        raise NotImplementedError("Different subscripts are not supported.")

    d = d.split_mode(m, math.gcd(*d.get_dims(m)))

    return d

@torch.no_grad()
def build_grouped_paths(path_segment_indices: torch.Tensor,
                        path_coefficients: torch.Tensor,
                        V: int,
                        device=None):
    """
    path_segment_indices: [P,4] int64/int32, columns: i,j,k,v
    path_coefficients:    [P]   float32/float64
    returns:
      i_list, j_list, k_list: [P] int32
      coeff_list:             [P] same dtype as coefficients
      v_offsets:              [V+1] int32, CSR-like offsets into lists
    """
    assert path_segment_indices.ndim == 2 and path_segment_indices.size(1) == 4
    P = path_segment_indices.size(0)
    if device is None:
        device = path_segment_indices.device

    psi = path_segment_indices.to("cpu")
    coeff = path_coefficients.to("cpu")

    v = psi[:, 3].to(torch.int64)
    # sort by v
    order = torch.argsort(v, stable=True)
    psi_s = psi[order]
    coeff_s = coeff[order]

    v_s = psi_s[:, 3].to(torch.int64)
    counts = torch.bincount(v_s, minlength=V)
    v_offsets = torch.zeros(V + 1, dtype=torch.int32)
    v_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)

    i_list = psi_s[:, 0].to(torch.int32).contiguous().to(device)
    j_list = psi_s[:, 1].to(torch.int32).contiguous().to(device)
    k_list = psi_s[:, 2].to(torch.int32).contiguous().to(device)
    v_list = psi_s[:, 3].to(torch.int32).contiguous().to(device)
    coeff_list = coeff_s.contiguous().to(device)
    v_offsets = v_offsets.contiguous().to(device)

    return i_list, j_list, k_list, v_list, coeff_list, v_offsets

@torch.no_grad()
def build_grouped_paths_as_buffers(
    module: nn.Module,
    path_segment_indices: torch.Tensor,
    path_coefficients: torch.Tensor,
    V: int,
    *,
    prefix: str = "",
    device=None,
    persistent: bool = True,
):
    """
    Build grouped path metadata and register them as buffers on `module`.

    Args:
        module: target nn.Module
        path_segment_indices: [P, 4], columns are (i, j, k, v)
        path_coefficients:    [P]
        V: number of output groups for CSR-like offsets
        prefix: optional buffer name prefix, e.g. "fwd_"
        device: target device; default is path_segment_indices.device
        persistent: whether buffers are saved in state_dict

    Registered buffers:
        {prefix}i_list      : [P] int32
        {prefix}j_list      : [P] int32
        {prefix}k_list      : [P] int32
        {prefix}v_list      : [P] int32
        {prefix}coeff_list  : [P] same dtype as path_coefficients
        {prefix}v_offsets   : [V+1] int32
    """
    assert path_segment_indices.ndim == 2 and path_segment_indices.size(1) == 4, \
        "path_segment_indices must have shape [P, 4]"

    if device is None:
        device = path_segment_indices.device

    psi = path_segment_indices.detach().to("cpu")
    coeff = path_coefficients.detach().to("cpu")

    # Sort by v
    v = psi[:, 3].to(torch.int64)
    order = torch.argsort(v, stable=True)
    psi_s = psi.index_select(0, order)
    coeff_s = coeff.index_select(0, order)

    # Build CSR-like offsets for v
    v_s = psi_s[:, 3].to(torch.int64)
    counts = torch.bincount(v_s, minlength=V)

    v_offsets = torch.zeros(V + 1, dtype=torch.int32)
    v_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)

    i_list = psi_s[:, 0].to(torch.int32).contiguous().to(device)
    j_list = psi_s[:, 1].to(torch.int32).contiguous().to(device)
    k_list = psi_s[:, 2].to(torch.int32).contiguous().to(device)
    v_list = psi_s[:, 3].to(torch.int32).contiguous().to(device)
    coeff_list = coeff_s.contiguous().to(device)
    v_offsets = v_offsets.contiguous().to(device)

    module.register_buffer(f"{prefix}i_list", i_list, persistent=persistent)
    module.register_buffer(f"{prefix}j_list", j_list, persistent=persistent)
    module.register_buffer(f"{prefix}k_list", k_list, persistent=persistent)
    module.register_buffer(f"{prefix}v_list", v_list, persistent=persistent)
    module.register_buffer(f"{prefix}coeff_list", coeff_list, persistent=persistent)
    module.register_buffer(f"{prefix}v_offsets", v_offsets, persistent=persistent)

@torch.no_grad()
def build_padded_stc_paths_as_buffers(
    module: nn.Module,
    path_segment_indices,
    path_coefficients,
    *,
    prefix: str = "stc_",
    device=None,
    persistent: bool = False,
    pad_value: int = STC_PAD_VALUE,
):
    """
    Build STC path metadata once and register it as module buffers.

    The generated STC-uniform1d kernel uses baseline-compatible padded path
    semantics:

        len == 3: [x1_a,                 x0_d, out_v, pad, ...]
        len == 4: [x1_a, x1_b,          x0_d, out_v, pad, ...]
        len == 5: [x1_a, x1_b, x1_c,   x0_d, out_v, pad, ...]

    We use ``pad_value=-1`` instead of 0 so that the valid prefix can be
    inferred from ``paths`` itself.  ``path_lens`` is still materialized once
    during preprocessing because the existing baseline/backward kernels use it,
    but forward does not need to recompute or parse it.

    Registered buffers:
        {prefix}coeffs      : [P], math dtype
        {prefix}paths       : [P, max_path_len], int32, padded with -1
        {prefix}path_lens   : [P], int32, inferred before padding
        {prefix}idx_lists   : [max_path_len, P], int32, padded with -1

    Returned metadata additionally contains:
        baseline_args       : args for fast_stc fallback
        uniform1d_jit_args  : args for fast_stc_uniform1d_jit, already parsed
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(path_segment_indices) == 0:
        raise ValueError("STC preprocessing got empty path_segment_indices")

    path_tensors = [
        torch.as_tensor(p, dtype=torch.int32).reshape(-1)
        for p in path_segment_indices
    ]
    path_lens = torch.tensor(
        [int(p.numel()) for p in path_tensors],
        dtype=torch.int32,
    )
    max_path_len = int(path_lens.max().item())
    if max_path_len < 3:
        raise ValueError(f"Bad STC max_path_len={max_path_len}; expected >= 3")

    # Pad variable-length paths to [P, max_path_len].  -1 is a true sentinel:
    # valid x1/x0/out segment indices are non-negative, so path_lens can be
    # recovered as (paths != -1).sum(dim=1) if needed.
    paths = pad_sequence(
        path_tensors,
        batch_first=True,
        padding_value=int(pad_value),
    ).contiguous()

    inferred_lens = (paths != int(pad_value)).sum(dim=1).to(torch.int32)
    if not torch.equal(inferred_lens.cpu(), path_lens.cpu()):
        raise RuntimeError("Internal STC padding error: inferred path_lens mismatch")

    # Ensure padding only appears on the right.  This prevents a malformed path
    # such as [x1, -1, x0, out, -1] from silently producing the wrong len.
    for pidx, L in enumerate(path_lens.tolist()):
        row = paths[pidx]
        if L > 0 and bool((row[:L] == int(pad_value)).any().item()):
            raise ValueError(f"STC path {pidx} contains pad_value inside valid prefix")
        if L < max_path_len and bool((row[L:] != int(pad_value)).any().item()):
            raise ValueError(f"STC path {pidx} has non-padding values after valid prefix")

    coeffs = torch.as_tensor(path_coefficients).contiguous()
    if coeffs.reshape(-1).numel() != len(path_tensors):
        raise ValueError(
            f"STC coeff count {coeffs.reshape(-1).numel()} does not match "
            f"num_paths={len(path_tensors)}"
        )

    coeffs_buf = coeffs.to(device=device)
    paths_buf = paths.to(device=device)
    path_lens_buf = path_lens.to(device=device)

    module.register_buffer(f"{prefix}coeffs", coeffs_buf, persistent=persistent)
    module.register_buffer(f"{prefix}paths", paths_buf, persistent=persistent)
    module.register_buffer(f"{prefix}path_lens", path_lens_buf, persistent=persistent)

    # List-based layout for STC-uniform1d adapter.  Shape is [max_path_len, P],
    # so idx_lists[t] is the t-th operand-index list across all paths.
    idx_lists_tensor = paths.t().contiguous()
    idx_lists_buf = idx_lists_tensor.to(device=device)
    module.register_buffer(f"{prefix}idx_lists", idx_lists_buf, persistent=persistent)

    paths_buffer = getattr(module, f"{prefix}paths")
    path_lens_buffer = getattr(module, f"{prefix}path_lens")
    coeffs_buffer = getattr(module, f"{prefix}coeffs")
    idx_lists_buffer = getattr(module, f"{prefix}idx_lists")

    meta = {
        "coeffs": coeffs_buffer,
        "paths": paths_buffer,
        "path_lens": path_lens_buffer,
        "idx_lists_tensor": idx_lists_buffer,
        "idx_lists": [idx_lists_buffer[t] for t in range(max_path_len)],
        "num_paths": int(len(path_segment_indices)),
        "max_path_len": max_path_len,
        "pad_value": int(pad_value),
    }

    # Pre-parse argument packs once.  Forward should only unpack these tuples;
    # it should not rebuild padding or call a flexible _parse_fast_stc_args path.
    meta["baseline_args"] = (
        coeffs_buffer,
        paths_buffer,
        path_lens_buffer,
    )
    meta["uniform1d_jit_args"] = (
        coeffs_buffer,
        paths_buffer,
        idx_lists_buffer,
    )

    return meta

@torch.no_grad()
def make_p_for_k(K_per_path, device):
    K_total = sum(K_per_path)
    p_for_k = torch.empty((K_total,), device=device, dtype=torch.int32)
    off = 0
    for p, kp in enumerate(K_per_path):
        p_for_k[off:off+kp] = p
        off += kp
    return p_for_k


@torch.no_grad()
def make_cg_single_mapping(K_total, I_total, device, path_num, math_dtype):
    """
    Build i_for_k[K], val_for_k[K], each k has at most one nnz (or empty).
      diag    : i_for_k[k]=k if k<I_total
      single0 : only k=0 uses i=0
    """
    i_for_k = torch.full((K_total,), -1, device=device, dtype=torch.int32)
    val_for_k = torch.zeros((K_total,), device=device, dtype=math_dtype)

    # "diag"
    if path_num == 4:
        kk = torch.arange(K_total, device=device, dtype=torch.int32)
        mask = kk < I_total
        i_for_k[mask] = kk[mask]
        val_for_k[mask] = 1.0
    elif path_num == 1:
        if K_total > 0:
            i_for_k[0] = 0
            val_for_k[0] = 1.0
    else:
        raise ValueError(f"Unknown path_num: {path_num}")

    return i_for_k, val_for_k

@torch.no_grad()
def infer_fctp_meta(descriptor, math_dtype, device):
    # per-path tensors
    cg_indices = []
    cg_values  = []
    c_tensors  = []
    dim_list   = []

    #print(f"i dims:", sum(descriptor.get_dims("i")))
    #print(f"j dims:", sum(descriptor.get_dims("j")))
    #print(f"k dims:", sum(descriptor.get_dims("k")))

    # 1) build per-path (idx, val, coeffs)
    for i, path in enumerate(descriptor.paths):
        if getattr(path, "coefficients", None) is None or path.coefficients.ndim < 3:
            raise ValueError("FCTP only supports paths with explicit 3D coefficient tensors.")

        coeffs = torch.from_numpy(path.coefficients).to(device=device, dtype=math_dtype)

        # idx: [nnz, 3] (i,j,k) ; vals: [nnz]
        idx = coeffs.nonzero(as_tuple=False).to(device=device, dtype=torch.int32)
        vals = coeffs[idx[:, 0], idx[:, 1], idx[:, 2]].to(device=device, dtype=math_dtype)

        dim_list.append(int(vals.numel()))

        cg_indices.append(idx)
        cg_values.append(vals)
        c_tensors.append(coeffs)

    # 2) global dimensions
    dimensions_dict = descriptor.get_dimensions_dict()
    U = sum(dimensions_dict["u"])
    V = sum(dimensions_dict["v"])
    W = sum(dimensions_dict["w"])

    # 3) K_per_path / offsets / totals
    P = len(cg_indices)
    assert P == len(dim_list)

    K_per_path = torch.tensor(dim_list, device=device, dtype=torch.int32)
    path_offset = torch.empty(P, device=device, dtype=torch.int32)
    path_offset[0] = 0
    if P > 1:
        path_offset[1:] = torch.cumsum(K_per_path[:-1], dim=0)
    
    #K_total = int(K_per_path.sum().item())
    I_total = sum(descriptor.get_dims("i"))
    K_total = sum(descriptor.get_dims("k"))


    # 4) pack nnz info
    nnz_list = [int(ci.shape[0]) for ci in cg_indices]
    nnz_max = max(nnz_list) if nnz_list else 0
    nnz_per_path = torch.tensor(nnz_list, device=device, dtype=torch.int32)

    # 5) pack cg_*_all
    cg_i_all   = torch.zeros((P, nnz_max), device=device, dtype=torch.int32)
    cg_j_all   = torch.zeros((P, nnz_max), device=device, dtype=torch.int32)
    cg_k_all   = torch.zeros((P, nnz_max), device=device, dtype=torch.int32)
    cg_val_all = torch.zeros((P, nnz_max), device=device, dtype=math_dtype)

    for p in range(P):
        ci_local = cg_indices[p]   # [nnz_p, 3]
        cv       = cg_values[p]    # [nnz_p]
        nnz_p    = nnz_list[p]
        offset_p = int(path_offset[p].item())

        i_local = ci_local[:, 0]
        j_local = ci_local[:, 1]
        k_local = ci_local[:, 2]

        # --- local -> global ---
        i_global = i_local + offset_p
        j_global = j_local              # TODO: only support j_local = j_global = 0
        k_global = k_local + offset_p

        cg_i_all[p, :nnz_p]   = i_global
        cg_j_all[p, :nnz_p]   = j_global
        cg_k_all[p, :nnz_p]   = k_global
        cg_val_all[p, :nnz_p] = cv
    
    i_for_k, val_for_k = make_cg_single_mapping(K_total, I_total, device, P, math_dtype)
    p_for_k = make_p_for_k(K_per_path, device)

    nnz0 = int(nnz_per_path[0])

    def can_use_empty_grad_x(i_for_k: torch.Tensor, I: int) -> bool:
        i_cpu = i_for_k.detach().cpu()
        valid = i_cpu[i_cpu >= 0]

        if valid.numel() != I:
            return False

        sorted_i = torch.sort(valid).values
        return torch.equal(sorted_i, torch.arange(I, dtype=sorted_i.dtype))
    
    print(f"cg_val_all:{cg_val_all}, shape:{cg_val_all.shape}", )

    # check that all non-zero CG coefficients are identical
    nonzero_cg_vals = cg_val_all[cg_val_all != 0]

    if nonzero_cg_vals.numel() == 0:
        raise ValueError("cg_val_all does not contain any non-zero CG coefficients.")

    cg_val_ref = nonzero_cg_vals[0]

    if not torch.allclose(
        nonzero_cg_vals,
        cg_val_ref.expand_as(nonzero_cg_vals),
        rtol=1e-5,
        atol=1e-8,
    ):
        unique_vals = torch.unique(nonzero_cg_vals.detach().cpu())
        raise ValueError(
            "FCTP requires all non-zero CG coefficients to be identical, "
            f"but found different values: {unique_vals.tolist()}"
        )

    cg_val_0 = cg_val_ref.item()

    return {
        "cg_indices": cg_indices,
        "cg_values": cg_values,
        "c_tensors": c_tensors,

        "U": U, "V": V, "W": W,

        "P": P,
        "K_per_path": K_per_path,
        "path_offset": path_offset,
        "I_total": I_total,
        "K_total": K_total,

        "nnz_list": nnz_list,
        "nnz_max": nnz_max,
        "nnz_per_path": nnz_per_path,

        "cg_i_all": cg_i_all,
        "cg_j_all": cg_j_all,
        "cg_k_all": cg_k_all,
        "cg_val_all": cg_val_all,
        "i_for_k": i_for_k,
        "val_for_k": val_for_k,
        "p_for_k": p_for_k,

        "nnz0": nnz0,
        "cg_val_0": cg_val_0,
        "can_use_empty_grad_x": can_use_empty_grad_x(i_for_k, I_total)
        
    }

class SegmentedPolynomial(nn.Module):
    """PyTorch module that computes a segmented polynomial.

    Args:
        polynomial: The segmented polynomial to compute, an instance of
            `cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`.
        method: Specifies the implementation method to use. Options are:

            - ``"naive"``: Uses a naive PyTorch implementation. It always works but is not optimized.
            - ``"uniform_1d"``: Uses a CUDA implementation for polynomials with a single uniform mode.
            - ``"fused_tp"``: Uses a CUDA implementation for polynomials with 3- and 4-operand contractions.
            - ``"indexed_linear"``: Uses a CUDA implementation for linear layers with indexed weights.

        math_dtype: Optional data type for computational operations.
            If specified, internal buffers will be of this dtype,
            and operands will be converted to this type for all computations.

            Values can be specified as a string corresponding to a torch.dtype,
            or as a torch.dtype.
            For some methods, special values can be used:

            - For method ``"naive"``: Any torch.dtype or corresponding string.
            - For method ``"uniform_1d"``: ``torch.float32`` or ``torch.float64`` or corresponding strings.
            - For method ``"fused_tp"``: ``torch.float32`` or ``torch.float64`` or corresponding strings.
            - For method ``"indexed_linear"``: this is not supported and will be ignored.

            .. note::
               This will not be affected by changes to the module dtype,
               and not all methods support all dtypes.

            If ``math_dtype`` is not specified:

            - For method ``"naive"``, the dtype of the input tensors will be used.
            - For method ``"uniform_1d"``, the dtype of the input tensors will be used if allowed
              (FP32 or FP64), otherwise float32 will be used.
            - For method ``"fused_tp"``, the default dtype (FP32) will be used.
            - For method ``"indexed_linear"``, the dtype of the input tensors will be used.

        output_dtype_map: Optional list that, for each output buffer, specifies
            the index of the input buffer from which it inherits its data type.
            -1 means the math_dtype is used.
            Default is 0 if there are input tensors, otherwise -1.
        name: Optional name for the operation. Defaults to "segmented_polynomial".

    Examples:
        Basic usage with spherical harmonics:

        >>> import torch
        >>> import cuequivariance as cue
        >>> from cuequivariance_torch import SegmentedPolynomial
        >>>
        >>> # Create spherical harmonics polynomial
        >>> poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2]).polynomial
        >>> sp = SegmentedPolynomial(poly, method="naive")
        >>>
        >>> # Compute spherical harmonics for unit vector along y-axis
        >>> x = torch.tensor([[0.0, 1.0, 0.0]])
        >>> result = sp([x])
        >>> print(result[0].shape)
        torch.Size([1, 9])

        Example with a linear layer:

        >>> # Create a linear transformation
        >>> input_irreps = cue.Irreps(cue.O3, "5x0e + 3x1o")
        >>> output_irreps = cue.Irreps(cue.O3, "4x0e + 2x1o")
        >>> poly = cue.descriptors.linear(input_irreps, output_irreps).polynomial
        >>>
        >>> # Create the module
        >>> linear = SegmentedPolynomial(poly, method="naive")
        >>>
        >>> # Create random weights and input
        >>> weights = torch.randn(1, poly.inputs[0].size)
        >>> x = torch.randn(10, poly.inputs[1].size)
        >>>
        >>> # Forward pass
        >>> result = linear([weights, x])
        >>> print(result[0].shape)
        torch.Size([10, 10])

        Example with indexed operations:

        >>> # Create indexed weights for different elements
        >>> weights = torch.randn(3, poly.inputs[0].size)  # 3 different weight sets
        >>> x = torch.randn(5, poly.inputs[1].size)        # 5 input vectors
        >>>
        >>> # Index tensor specifying which weights to use for each input
        >>> weight_indices = torch.tensor([0, 1, 0, 2, 1])  # Use weights 0,1,0,2,1
        >>>
        >>> result = linear([weights, x],
        ...                input_indices={0: weight_indices})
        >>> print(result[0].shape)
        torch.Size([5, 10])
    """

    def __init__(
        self,
        polynomial: cue.SegmentedPolynomial,
        method: str = "",
        math_dtype: str | torch.dtype = None,
        output_dtype_map: List[int] = None,
        name: str = "segmented_polynomial",
        op_name: str = "",
        use_fasteq: bool = True,
        u1d_compatible: bool = False,
    ):
        super().__init__()

        self.num_inputs = polynomial.num_inputs
        self.num_outputs = polynomial.num_outputs
        self.method = method
        self.repr = polynomial.__repr__()
        self.op_name = op_name
        self.descriptor = polynomial.operations[0][1]
        self.use_fasteq = use_fasteq
        self.polynomial = polynomial
        self.u1d_compatible = u1d_compatible
        
        if method == "":
            warnings.warn(
                "Hello! It looks like you're using code that was written for an older version of this library.\n"
                "Starting in v0.6.0, the `method` argument is suggested when using `SegmentedPolynomial()`.\n"
                "This change helps ensure you get optimal performance by explicitly choosing the computation method.\n"
                "For the moment, we will default to the 'uniform_1d' method.\n\n"
                "To remove this warning, add a `method` parameter to your function call. Here are the available options:\n"
                "• 'naive' - Works everywhere but not optimized (good for testing)\n"
                "• 'uniform_1d' - Fast CUDA implementation for single uniform mode polynomials\n"
                "• 'fused_tp' - A more general CUDA implementation, supporting many 3 and 4 operands contractions.\n"
                "• 'indexed_linear' - A CUDA implementation for linear layers with indexed weights.\n"
            )
            method = "naive"

        if not isinstance(polynomial, cue.SegmentedPolynomial):
            raise ValueError(
                f"The polynomial is not a cue.SegmentedPolynomial, but a {type(polynomial)}",
                "Did you forget to call `.polynomial` on the descriptor?",
            )

        if method == "uniform_1d":
            self.m = SegmentedPolynomialFromUniform1dJit(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = self.m
        elif method == "naive":
            self.m = SegmentedPolynomialNaive(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = self.m
        elif method == "fused_tp":
            self.m = SegmentedPolynomialFusedTP(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = SegmentedPolynomialNaive(
                polynomial, math_dtype, output_dtype_map, name
            )
        elif method == "indexed_linear":
            self.m = SegmentedPolynomialIndexedLinear(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = self.m
        else:
            raise ValueError(f"Invalid method: {method}")

        if use_fasteq and op_name == "stc" and self.method == "uniform_1d":
            ds_ = [flatten_stp(d) for _, d in polynomial.operations]
            d_max = max(ds_, key=lambda d: d.num_operands)
            self.num_out_segments = d_max.operands[-1].num_segments
            self.u = d_max.operands[0].size // d_max.operands[0].num_segments

            path_segment_indices = sum((d.indices.tolist() for d in ds_), [])
            path_coefficients = sum((d.stacked_coefficients.tolist() for d in ds_), [])

            print(f"op_name:{op_name}, desc:{self.descriptor}") 


            self.stc_meta = build_padded_stc_paths_as_buffers(
                self,
                path_segment_indices,
                torch.as_tensor(path_coefficients, dtype=math_dtype),
                prefix="stc_",
                device="cuda",
                persistent=False,
                pad_value=STC_PAD_VALUE,
            )
            self.stc_meta["num_out_segments"] = int(self.num_out_segments)
            self.stc_meta["u"] = int(self.u)
            self.stc_meta["uniform1d_jit_args"] = (
                self.stc_meta["coeffs"],
                self.stc_meta["paths"],
                self.stc_meta["idx_lists_tensor"],
                self.stc_meta["num_out_segments"],
            )
            #print(f"stc meta:{self.stc_meta}")
        
        elif use_fasteq and (self.op_name == "cwtp" or self.method == "uniform_1d"):

            self.descriptor = self.m.polynomial.operations[0][1]
            print(f"uniform1d path, descriptor:{self.descriptor}")
            print(f"polynomial.operations:{self.m.polynomial.operations}")

            ds_ = [d for _, d in self.m.polynomial.operations]
            self.path_indices = sum((d.indices.tolist() for  d in ds_), [])
            self.path_coefficients = sum((d.stacked_coefficients.tolist() for d in ds_), [])
            self.num_segments_list = [ operand.num_segments for operand in self.descriptor.operands]
            self.size_list = [ operand.size for operand in self.descriptor.operands]

            path_segment_indices_tensor = torch.tensor(self.path_indices, dtype=torch.int32, device="cuda")
            path_coefficients_tensor = torch.tensor(self.path_coefficients, dtype=math_dtype, device="cuda")
            u_dim = list(self.descriptor.get_dims("u"))[0]
            w_seg_num, x_seg_num, y_seg_num, out_seg_num = self.num_segments_list[0], self.num_segments_list[1], self.num_segments_list[2], self.num_segments_list[3]
            build_grouped_paths_as_buffers(self, path_segment_indices_tensor, path_coefficients_tensor, out_seg_num, device="cuda", persistent=False)
            self.u1d_meta = {}

            self.u1d_meta["i_list"] = self.i_list
            self.u1d_meta["j_list"] = self.j_list
            self.u1d_meta["k_list"] = self.k_list
            self.u1d_meta["v_list"] = self.v_list
            self.u1d_meta["coeff_list"] = self.coeff_list
            self.u1d_meta["size_list"] = self.size_list
            self.u1d_meta["v_offsets"] = self.v_offsets
            self.u1d_meta["out_seg_num"] = out_seg_num
            self.u1d_meta["w_seg_num"] = w_seg_num
            self.u1d_meta["x_seg_num"] = x_seg_num
            self.u1d_meta["y_seg_num"] = y_seg_num
            self.u1d_meta["u_dim"] = u_dim
            
        
        elif use_fasteq and (op_name == "fctp"):
            self.meta = infer_fctp_meta(self.descriptor, math_dtype=math_dtype, device="cuda")
            


    def __repr__(self):
        return self.repr + f"\n{super().__repr__()}"

    # For torch.jit.trace, we cannot pass explicit optionals,
    # so these must be passed as kwargs then.
    # List[Optional[Tensor]] does not work for similar reasons, hence, Dict
    # is the only option.
    # Also, shapes cannot be passed as integers, so they are passed via a
    # (potentially small-strided) tensor with the right shape.
    def forward(
        self,
        inputs: List[torch.Tensor],
        input_indices: Optional[Dict[int, torch.Tensor]] = None,
        output_shapes: Optional[Dict[int, torch.Tensor]] = None,
        output_indices: Optional[Dict[int, torch.Tensor]] = None,
    ):
        """Compute the segmented polynomial based on the specified descriptor.

        Args:
            inputs: The input tensors. The number of input tensors must match
                the number of input buffers in the descriptor.
                Each input tensor should have a shape of ``(batch, operand_size)`` or
                ``(1, operand_size)`` or ``(index, operand_size)`` in the indexed case.
                Here, ``operand_size`` is the size of each operand as defined in
                the descriptor.
            input_indices: A dictionary that contains an optional indexing tensor
                for each input tensor. The key is the index into the inputs.
                If a key is not present, no indexing takes place.
                The contents of the index tensor must be suitable to index the
                input tensor (i.e., ``0 <= index_tensor[i] < input.shape[0]``).

                .. note::
                   Method ``"indexed_linear"`` requires the indices to be sorted.

            output_shapes: A dictionary specifying the size of the output batch
                dimensions using Tensors. We only read ``shape_tensor.shape[0]``.
                This is mandatory if the output tensor is indexed. Otherwise,
                the default shape is ``(batch, operand_size)``.
            output_indices: A dictionary that contains an optional indexing tensor
                for each output tensor. See ``input_indices`` for details.

        Returns:
            The output tensors resulting from the segmented polynomial.
            Their shapes are specified just like the inputs.
        """
        #print(f"op name:{self.op_name}, polynomial.operations:{self.polynomial.operations}")
        # General checks
        empty_dict: Dict[int, torch.Tensor] = {}
        if input_indices is None:
            input_indices = dict(empty_dict)
        if output_shapes is None:
            output_shapes = dict(empty_dict)
        if output_indices is None:
            output_indices = dict(empty_dict)

        inputs = list(inputs)

        if not torch.jit.is_scripting():
            if (
                not torch.jit.is_tracing()
                and not torch.compiler.is_compiling()
                and not torch.fx._symbolic_trace.is_fx_tracing()
            ):
                torch._assert(
                    len(inputs) == self.num_inputs,
                    "the number of inputs must match the number of inputs of the polynomial",
                )

                for k, v in input_indices.items():
                    torch._assert(
                        0 <= k < self.num_inputs, "input index must be in range"
                    )
                    torch._assert(v.ndim == 1, "input index must be one-dimensional")
                    torch._assert(
                        v.dtype in [torch.int32, torch.int64],
                        "input index must be integral",
                    )
                for k, v in output_indices.items():
                    torch._assert(
                        0 <= k < self.num_outputs, "output index must be in range"
                    )
                    torch._assert(v.ndim == 1, "input index must be one-dimensional")
                    torch._assert(
                        v.dtype in [torch.int32, torch.int64],
                        "input index must be integral",
                    )
                for k, v in output_shapes.items():
                    torch._assert(
                        0 <= k < self.num_outputs, "output index must be in range"
                    )
                    torch._assert(v.ndim == 2, "output shape must be two-dimensional")

                # If the input is on the CPU and we're using fused_tp, we need to fall back to naive
                if (
                    inputs[0].device == torch.device("cpu")
                    and self.method == "fused_tp"
                ):
                    warnings.warn(
                        "Fused TP is not supported on CPU. Falling back to naive implementation."
                    )
                    return self.fallback(
                        inputs, input_indices, output_shapes, output_indices
                    )
        out = [torch.empty(0) for _ in range(self.num_outputs)]
        #print(f"=========== use fasteq: {self.use_fasteq}, op name: {self.op_name} ==============")
        if self.use_fasteq:
            if self.op_name == "equi_linear":
                if self.num_outputs != 1:
                    raise ValueError("equi_linear should have exactly one output")
                ref = fast_equi_linear(self.descriptor, inputs[0], inputs[1])
                out[0] = ref
                #out = self.m(inputs, input_indices, output_shapes, output_indices)
            elif self.op_name == "stc":
                i0 = input_indices[0].to(torch.int32)
                x0 = inputs[0]
                x1 = inputs[1]
                
                x0 = x0.reshape(x0.shape[0], x0.shape[1] // self.u, self.u)
                x1 = x1.reshape(x1.shape[0], x1.shape[1] // self.u, self.u)

                #print(f"x1 shape:{x1.shape}, x0 shape:{x0.shape}, i0 shape:{i0.shape}")

                ref = fast_stc_uniform1d_jit(
                    x1, x0, i0,
                    *self.stc_meta["uniform1d_jit_args"],
                )
                
                out[0] = ref
            elif self.method == "uniform_1d":
                
                w = inputs[0]
                x = inputs[1]
                y = inputs[2]
                
                ref = fast_uniform1d_jit(w, x, y, input_indices, output_indices, self.u1d_meta)
                out[0] = ref
            elif self.op_name == "fctp":
                w, x, y = inputs[0], inputs[1], inputs[2]
                ref = fast_fctp(
                    w, x, y,
                    self.meta,
                )
                out[0] = ref
            else:
                out = self.m(inputs, input_indices, output_shapes, output_indices)
                
        else:
            out = self.m(inputs, input_indices, output_shapes, output_indices)
        return out
