"""Quantized matmul kernels: Triton on CUDA / ROCm, pure torch elsewhere.

Decode (one or a few rows) never materializes weights: Q8_0 matrix-vector products read
the int8 matrix and its per-32 scales directly, and the IQ3_XXS mixture-of-experts kernels
decode only the 8 selected experts' rows while multiplying. Prefill (many rows) dequantizes
a weight to the compute dtype and uses a dense matmul.

IQ3_XXS kernels work on groups of 4 weights: one grid lookup gives 4 magnitudes, one
32-bit word per 32 weights gives the scale and four 7-bit sign indices (the 8th sign bit is
even parity, as in ggml's ksigns_iq2xs).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .quants import IQ3_BLOCK_BYTES, iq3_dequant_torch, iq3_grid, q8_dequant_torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = os.environ.get("DRAMA3_NO_TRITON", "0") != "1"
except Exception:  # noqa: BLE001 - any import failure means no Triton
    triton = None
    HAS_TRITON = False


def use_triton(t: torch.Tensor) -> bool:
    return HAS_TRITON and t.is_cuda


class StaticArena:
    """Buffers for every tensor a Triton kernel touches inside a CUDA-graph capture.

    Under PyTorch's cudaMallocAsync allocator (ComfyUI's default), memory allocated during
    stream capture has no address yet, and Triton's launcher rejects it. While active, the
    arena hands out buffers in call order: allocated on the first (uncaptured) warm-up run,
    then the same buffers on every later run, including captures. Each run makes the same
    calls in the same order, so buffer i always serves the same call site.
    """

    def __init__(self) -> None:
        self.buffers: list[torch.Tensor] = []
        self.index = 0
        self.active = False
        self.copy_inputs = False

    def begin(self) -> None:
        # Kernel inputs computed during capture need static copies only under cudaMallocAsync;
        # the default caching allocator gives capture-time tensors real addresses.
        self.active, self.index = True, 0
        self.copy_inputs = torch.cuda.get_allocator_backend() == "cudaMallocAsync"

    def end(self) -> None:
        self.active = False

    def empty(self, shape, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if not self.active:
            return torch.empty(shape, dtype=dtype, device=device)
        shape = tuple(shape)
        if self.index < len(self.buffers):
            t = self.buffers[self.index]
            if t.shape != shape or t.dtype != dtype:
                raise RuntimeError(f"static arena call {self.index}: {tuple(t.shape)} {t.dtype} != {shape} {dtype}")
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("static arena: new buffer requested during capture; warm up first")
            t = torch.empty(shape, dtype=dtype, device=device)
            self.buffers.append(t)
        self.index += 1
        return t

    def input(self, x: torch.Tensor) -> torch.Tensor:
        """x itself outside the arena; otherwise a static copy."""
        if not (self.active and self.copy_inputs):
            return x.contiguous()
        buf = self.empty(x.shape, x.dtype, x.device)
        buf.copy_(x)
        return buf

    def clear(self) -> None:
        self.buffers.clear()
        self.index = 0


ARENA = StaticArena()  # the decoder that is warming up / capturing installs its own arena here


if HAS_TRITON:

    @triton.jit
    def _q8_gemv_kernel(x_ptr, q_ptr, d_ptr, o_ptr, N, K, sxm, sqn, sdn, som,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_n = tl.program_id(0)
        m = tl.program_id(1)
        rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        rows64 = rows.to(tl.int64)
        G: tl.constexpr = BLOCK_K // 32
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            cols = k0 + tl.arange(0, BLOCK_K)
            q = tl.load(q_ptr + rows64[:, None] * sqn + cols[None, :], mask=rmask[:, None], other=0).to(tl.float32)
            x = tl.load(x_ptr + m * sxm + cols).to(tl.float32)
            s = tl.sum(tl.reshape(q * x[None, :], (BLOCK_N, G, 32)), axis=2)
            g = k0 // 32 + tl.arange(0, G)
            d = tl.load(d_ptr + rows64[:, None] * sdn + g[None, :], mask=rmask[:, None], other=0).to(tl.float32)
            acc += tl.sum(s * d, axis=1)
        tl.store(o_ptr + m * som + rows, acc, mask=rmask)

    @triton.jit
    def _q8_dequant_kernel(q_ptr, d_ptr, o_ptr, n_groups, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        g = pid * BLOCK + tl.arange(0, BLOCK)
        gmask = g < n_groups
        g64 = g.to(tl.int64)
        d = tl.load(d_ptr + g64, mask=gmask, other=0).to(tl.float32)
        c = tl.arange(0, 32)
        q = tl.load(q_ptr + g64[:, None] * 32 + c[None, :], mask=gmask[:, None], other=0).to(tl.float32)
        tl.store(o_ptr + g64[:, None] * 32 + c[None, :], (q * d[:, None]).to(o_ptr.dtype.element_ty), mask=gmask[:, None])

    @triton.jit
    def _iq3_block(ptr, base, grid_ptr):
        """Decode one 256-weight IQ3_XXS block for each row: base [R, 1] byte offsets -> fp32 [R, 256]."""
        grp = tl.arange(0, 64)[None, :]  # group of 4 weights: 8 * ib32 + 2 * l + half
        t = tl.arange(0, 4)[None, None, :]
        d_lo = tl.load(ptr + base).to(tl.uint16)
        d_hi = tl.load(ptr + base + 1).to(tl.uint16)
        d = (d_lo | (d_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)  # [R, 1]
        qidx = tl.load(ptr + base + 2 + grp).to(tl.int32)  # [R, 64]
        s_off = base + 66 + (grp // 8) * 4
        aux = (tl.load(ptr + s_off).to(tl.uint32)
               | (tl.load(ptr + s_off + 1).to(tl.uint32) << 8)
               | (tl.load(ptr + s_off + 2).to(tl.uint32) << 16)
               | (tl.load(ptr + s_off + 3).to(tl.uint32) << 24))  # [R, 64]
        db = d * (0.5 + (aux >> 28).to(tl.float32)) * 0.5
        l = (grp % 8) // 2
        sidx = (aux >> (7 * l).to(tl.uint32)) & 127
        p = sidx ^ (sidx >> 4)
        p = p ^ (p >> 2)
        p = p ^ (p >> 1)
        signs = (sidx | ((p & 1) << 7)).to(tl.int32)  # ksigns_iq2xs
        half = (grp % 2)[:, :, None]
        neg = (signs[:, :, None] >> (half * 4 + t)) & 1  # [R, 64, 4]
        g = tl.load(grid_ptr + qidx)  # [R, 64] packed magnitudes
        mag = ((g[:, :, None] >> (8 * t)) & 0xFF).to(tl.float32)
        y = db[:, :, None] * mag * (1.0 - 2.0 * neg.to(tl.float32))
        return tl.reshape(y, (base.shape[0], 256))

    @triton.jit
    def _iq3_gate_up_kernel(x_ptr, ids_ptr, g_ptr, u_ptr, grid_ptr, h_ptr, I, sxm, row_bytes, expert_bytes,
                            TOPK: tl.constexpr, NB: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_n = tl.program_id(0)
        pair = tl.program_id(1)
        m = pair // TOPK
        e = tl.load(ids_ptr + pair).to(tl.int64)
        rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_base = (e * expert_bytes + rows.to(tl.int64) * row_bytes)[:, None]
        w = tl.arange(0, 256)
        acc_g = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for b in range(NB):
            x = tl.load(x_ptr + m * sxm + b * 256 + w).to(tl.float32)[None, :]
            acc_g += tl.sum(_iq3_block(g_ptr, row_base + b * 98, grid_ptr) * x, axis=1)
            acc_u += tl.sum(_iq3_block(u_ptr, row_base + b * 98, grid_ptr) * x, axis=1)
        h = acc_g * tl.sigmoid(acc_g) * acc_u
        tl.store(h_ptr + pair * I + rows, h)

    @triton.jit
    def _iq3_down_kernel(h_ptr, ids_ptr, wts_ptr, d_ptr, grid_ptr, o_ptr, H, I, row_bytes, expert_bytes,
                         TOPK: tl.constexpr, NB: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_n = tl.program_id(0)
        m = tl.program_id(1)
        rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        w = tl.arange(0, 256)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for s in range(TOPK):
            pair = m * TOPK + s
            e = tl.load(ids_ptr + pair).to(tl.int64)
            weight = tl.load(wts_ptr + pair).to(tl.float32)
            row_base = (e * expert_bytes + rows.to(tl.int64) * row_bytes)[:, None]
            part = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for b in range(NB):
                hx = tl.load(h_ptr + pair * I + b * 256 + w)[None, :]
                part += tl.sum(_iq3_block(d_ptr, row_base + b * 98, grid_ptr) * hx, axis=1)
            acc += weight * part
        tl.store(o_ptr + m * H + rows, acc)

    @triton.jit
    def _iq3_down_split_kernel(h_ptr, ids_ptr, d_ptr, grid_ptr, o_ptr, H, I, row_bytes, expert_bytes,
                               NB: tl.constexpr, BLOCK_N: tl.constexpr):
        """One (token, expert) pair per program column: unweighted partial outputs [pairs, H]."""
        pid_n = tl.program_id(0)
        pair = tl.program_id(1)
        rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        w = tl.arange(0, 256)
        e = tl.load(ids_ptr + pair).to(tl.int64)
        row_base = (e * expert_bytes + rows.to(tl.int64) * row_bytes)[:, None]
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for b in range(NB):
            hx = tl.load(h_ptr + pair * I + b * 256 + w)[None, :]
            acc += tl.sum(_iq3_block(d_ptr, row_base + b * 98, grid_ptr) * hx, axis=1)
        tl.store(o_ptr + pair * H + rows, acc)

    @triton.jit
    def _iq3_dequant_kernel(q_ptr, grid_ptr, o_ptr, n_blocks, BLOCKS: tl.constexpr):
        pid = tl.program_id(0)
        blk = pid * BLOCKS + tl.arange(0, BLOCKS)
        valid = blk < n_blocks
        blk = tl.where(valid, blk, 0).to(tl.int64)
        y = _iq3_block(q_ptr, (blk * 98)[:, None], grid_ptr)
        w = tl.arange(0, 256)
        tl.store(o_ptr + blk[:, None] * 256 + w[None, :], y.to(o_ptr.dtype.element_ty), mask=valid[:, None])


# ---------------------------------------------------------------------------
# Q8_0


def q8_dequant(qs: torch.Tensor, d: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if not use_triton(qs):
        return q8_dequant_torch(qs, d, dtype)
    out = torch.empty(qs.shape, dtype=dtype, device=qs.device)
    n_groups = d.numel()
    BLOCK = 64
    _q8_dequant_kernel[(triton.cdiv(n_groups, BLOCK),)](qs, d, out, n_groups, BLOCK=BLOCK)
    return out


GEMV_MAX_ROWS = 4
MOE_BLOCK_N = int(os.environ.get("DRAMA3_MOE_BLOCK_N", "16"))
# Rows per program when the expert banks sit in pinned host memory (expert offload): the kernel then reads them over PCIe, where a
# larger tile moves more bytes per round trip. The tile only changes how rows are grouped, not the arithmetic of a row.
MOE_BLOCK_N_HOST = int(os.environ.get("DRAMA3_MOE_BLOCK_N_HOST", "32"))
MOE_DOWN_SPLIT = os.environ.get("DRAMA3_MOE_DOWN_SPLIT", "0") == "1"


def q8_linear(x: torch.Tensor, qs: torch.Tensor, d: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """x [..., K] @ W[N, K]^T for a Q8_0 weight; fp32 accumulation."""
    out_dtype = out_dtype or x.dtype
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    rows, (n, k) = x2.shape[0], qs.shape
    if use_triton(qs) and rows <= GEMV_MAX_ROWS:
        x2 = ARENA.input(x2)
        out = ARENA.empty((rows, n), torch.float32, x.device)
        BLOCK_N = 16 if n >= 2048 else 8
        _q8_gemv_kernel[(triton.cdiv(n, BLOCK_N), rows)](
            x2, qs, d, out, n, k, x2.stride(0), qs.stride(0), d.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N, BLOCK_K=256, num_warps=4)
        return out.to(out_dtype).reshape(*lead, n)
    w = q8_dequant(qs, d, x.dtype)
    if out_dtype == torch.float32 and x.dtype != torch.float32:
        out = _mm_fp32_out(x2, w)  # keep fp32 sums: these feed the fp32 residual (and fp16 can overflow)
    else:
        out = F.linear(x2, w).to(out_dtype)
    return out.reshape(*lead, n)


_MM_OUT_DTYPE = True


def _mm_fp32_out(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x @ w.T with fp32 output from half-precision inputs: torch.mm(out_dtype=) where available
    (PyTorch 2.8+ on CUDA), else an fp32 matmul."""
    global _MM_OUT_DTYPE
    if _MM_OUT_DTYPE and x.is_cuda:
        try:
            return torch.mm(x, w.T, out_dtype=torch.float32)
        except (TypeError, RuntimeError):
            _MM_OUT_DTYPE = False
    return F.linear(x.float(), w.float())


# ---------------------------------------------------------------------------
# IQ3_XXS


def iq3_dequant(raw: torch.Tensor, cols: int, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    """Packed rows [..., cols/256*98] -> dtype [..., cols] on `device` (default: raw's device).

    Pinned host rows (expert offload) are read in place by the GPU kernel over PCIe."""
    device = torch.device(device) if device is not None else raw.device
    host = not raw.is_cuda
    if not (HAS_TRITON and device.type == "cuda" and (not host or raw.is_pinned())):
        return iq3_dequant_torch(raw, cols, dtype).to(device)
    lead = raw.shape[:-1]
    flat = raw.reshape(-1)
    n_blocks = flat.numel() // IQ3_BLOCK_BYTES
    out = torch.empty(n_blocks * 256, dtype=dtype, device=device)
    BLOCKS = 8
    _iq3_dequant_kernel[(triton.cdiv(n_blocks, BLOCKS),)](flat, iq3_grid(device), out, n_blocks, BLOCKS=BLOCKS)
    return out.view(*lead, cols)


def iq3_moe(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
            gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """Routed experts for a few rows: x [M, H], ids / weights [M, k] -> fp32 [M, H].

    gate / up: packed [E, I, H/256*98]; down: packed [E, H, I/256*98].
    """
    m, hidden = x.shape
    top_k = ids.shape[1]
    inter = gate.shape[1]
    if use_triton(x):
        x = ARENA.input(x)
        ids32 = ARENA.input(ids.to(torch.int32))
        weights = ARENA.input(weights.float())
        grid = iq3_grid(x.device)
        h = ARENA.empty((m * top_k, inter), torch.float32, x.device)
        BLOCK_N = MOE_BLOCK_N if gate.is_cuda else MOE_BLOCK_N_HOST
        _iq3_gate_up_kernel[(inter // BLOCK_N, m * top_k)](
            x, ids32, gate, up, grid, h, inter, x.stride(0), gate.shape[2], gate.shape[1] * gate.shape[2],
            TOPK=top_k, NB=hidden // 256, BLOCK_N=BLOCK_N, num_warps=4)
        if MOE_DOWN_SPLIT:
            part = ARENA.empty((m * top_k, hidden), torch.float32, x.device)
            _iq3_down_split_kernel[(hidden // BLOCK_N, m * top_k)](
                h, ids32, down, grid, part, hidden, inter, down.shape[2], down.shape[1] * down.shape[2],
                NB=inter // 256, BLOCK_N=BLOCK_N, num_warps=4)
            return (part.view(m, top_k, hidden) * weights.unsqueeze(-1)).sum(1)
        out = ARENA.empty((m, hidden), torch.float32, x.device)
        _iq3_down_kernel[(hidden // BLOCK_N, m)](
            h, ids32, weights, down, grid, out, hidden, inter, down.shape[2],
            down.shape[1] * down.shape[2], TOPK=top_k, NB=inter // 256, BLOCK_N=BLOCK_N, num_warps=4)
        return out
    flat = ids.reshape(-1)
    wg = iq3_dequant(gate[flat], hidden, x.dtype)  # [M*k, I, H]
    wu = iq3_dequant(up[flat], hidden, x.dtype)
    wd = iq3_dequant(down[flat], inter, x.dtype)  # [M*k, H, I]
    rows = x.repeat_interleave(top_k, dim=0).unsqueeze(-1)
    g = torch.bmm(wg, rows).squeeze(-1).float()
    u = torch.bmm(wu, rows).squeeze(-1).float()
    h = (F.silu(g) * u).to(x.dtype).unsqueeze(-1)
    out = torch.bmm(wd, h).squeeze(-1).float() * weights.reshape(-1, 1).float()
    return out.view(m, top_k, hidden).sum(1)
