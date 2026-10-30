"""General FastEq FCTP: sparse multilinear contraction with segment offsets."""
import torch
import triton
import triton.language as tl


@triton.jit
def _contract(a, b, c, ia, ib, ic, io, coefficients, out,
              N: tl.constexpr, A: tl.constexpr, B: tl.constexpr,
              C: tl.constexpr, O: tl.constexpr, SA: tl.constexpr,
              SB: tl.constexpr, SC: tl.constexpr, DOUBLE: tl.constexpr,
              BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    t = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = t < N
    ai = tl.load(ia + t, valid, 0)
    bi = tl.load(ib + t, valid, 0)
    ci = tl.load(ic + t, valid, 0)
    oi = tl.load(io + t, valid, 0)
    dtype = tl.float64 if DOUBLE else tl.float32
    av = tl.load(a + batch * SA + ai, valid, 0).to(dtype)
    bv = tl.load(b + batch * SB + bi, valid, 0).to(dtype)
    cv = tl.load(c + batch * SC + ci, valid, 0).to(dtype)
    coeff = tl.load(coefficients + t, valid, 0).to(dtype)
    tl.atomic_add(out + batch * O + oi, av * bv * cv * coeff, valid)


class _SparseProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c, ia, ib, ic, io, coefficients, size):
        if not all(t.is_cuda for t in (a, b, c)):
            raise ValueError("FastEq FCTP requires CUDA tensors")
        original_inputs = (a, b, c)
        a, b, c = a.contiguous(), b.contiguous(), c.contiguous()
        batch = max(a.shape[0], b.shape[0], c.shape[0])
        if any(t.shape[0] not in (1, batch) for t in (a, b, c)):
            raise ValueError("FCTP batch dimensions must match or be one")
        out = a.new_zeros((batch, size))
        if io.numel():
            _contract[(batch, triton.cdiv(io.numel(), 256))](
                a, b, c, ia, ib, ic, io, coefficients, out,
                io.numel(), a.shape[1], b.shape[1], c.shape[1], size,
                0 if a.shape[0] == 1 else a.stride(0),
                0 if b.shape[0] == 1 else b.stride(0),
                0 if c.shape[0] == 1 else c.stride(0),
                a.dtype == torch.float64, 256,
            )
        ctx.save_for_backward(*original_inputs, ia, ib, ic, io, coefficients)
        return out

    @staticmethod
    def backward(ctx, grad):
        a, b, c, ia, ib, ic, io, coeff = ctx.saved_tensors
        grads = []
        # The adjoints are the same sparse multilinear contraction with the
        # output index exchanged with the differentiated input index. This
        # also supplies mixed second derivatives without detached recomputation.
        for needed, target, other1, other2, it, i1, i2 in (
            (ctx.needs_input_grad[0], a, b, c, ia, ib, ic),
            (ctx.needs_input_grad[1], b, a, c, ib, ia, ic),
            (ctx.needs_input_grad[2], c, a, b, ic, ia, ib),
        ):
            grads.append(_SparseProduct.apply(
                grad, other1, other2, io, i1, i2, it, coeff, target.shape[1]
            ).sum_to_size(target.shape) if needed else None)
        return (*grads, None, None, None, None, None, None)


def fast_fctp(w, x, y, meta):
    dtype = meta["coefficients"].dtype
    out = _SparseProduct.apply(
        w.to(dtype), x.to(dtype), y.to(dtype),
        *meta["indices"], meta["coefficients"], meta["output_size"]
    )
    return out.to(x.dtype)
