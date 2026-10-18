"""Check the engine without ComfyUI: dequant kernels vs the pure-torch reference, then one generated
line (no reference) and one cloned from it. Writes WAVs next to --out.

usage: python tools/selftest.py path/to/drama3-small-IQ3_XXS.gguf [--device cuda] [--out selftest]
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drama3_torch import kernels  # noqa: E402
from drama3_torch.generate import Drama3TTS, Sampling  # noqa: E402
from drama3_torch.text import Take  # noqa: E402


def write_wav(path: Path, audio: torch.Tensor, rate: int) -> None:
    pcm = (audio.clamp(-1, 1).numpy() * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="selftest")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    print(f"torch {torch.__version__}, device {args.device}, triton kernels: {kernels.HAS_TRITON}")
    tts = Drama3TTS(args.gguf, args.device)
    print(f"loaded in {tts.load_seconds:.1f}s, dtype {tts.model.dtype}")

    if kernels.use_triton(tts.model.backbone.head.qs):
        moe = tts.model.backbone.layers[3][4]
        raw, cols = moe.banks.gate[:2], moe.banks.gate.shape[2] // 98 * 256
        fast = kernels.iq3_dequant(raw, cols, torch.float32)
        kernels.HAS_TRITON = False
        ref = kernels.iq3_dequant(raw, cols, torch.float32)
        kernels.HAS_TRITON = True
        print("IQ3_XXS Triton dequant matches torch:", torch.equal(fast, ref))

        # The expert kernel must give the same bits for every tile size, with the banks in VRAM or in pinned host memory.
        x = torch.randn(1, moe.router.shape[1], generator=torch.Generator().manual_seed(0)).to(tts.model.device, tts.model.dtype)
        experts = moe.banks.gate.shape[0]  # differs per layer
        ids = torch.randperm(experts, generator=torch.Generator().manual_seed(1))[:8].view(1, 8).to(tts.model.device)
        weights = torch.full((1, 8), 1 / 8, device=tts.model.device)
        host = [t.cpu().pin_memory() for t in (moe.banks.gate, moe.banks.up, moe.banks.down)]
        saved = kernels.MOE_BLOCK_N, kernels.MOE_BLOCK_N_HOST
        outputs = []
        for tile in (8, 16, 32):
            kernels.MOE_BLOCK_N = kernels.MOE_BLOCK_N_HOST = tile
            outputs.append(kernels.iq3_moe(x, ids, weights, moe.banks.gate, moe.banks.up, moe.banks.down).clone())
            outputs.append(kernels.iq3_moe(x, ids, weights, *host).clone())
        kernels.MOE_BLOCK_N, kernels.MOE_BLOCK_N_HOST = saved
        print("iq3_moe identical for tiles 8/16/32, VRAM and host banks:", all(torch.equal(outputs[0], o) for o in outputs[1:]))

    t = time.time()
    first = tts.generate("[a gentle, soft-spoken woman's voice] Hello! This is a quick test of drama3-small.",
                         sampling=Sampling(seed=7))
    print(f"generated voice: {len(first.audio) / first.sample_rate:.1f}s audio in {time.time() - t:.1f}s {first.chunks}")
    write_wav(out / "generated.wav", first.audio, first.sample_rate)
    t = time.time()
    kept = tts.generate("And this second line keeps the same voice.",
                        [Take("Hello! This is a quick test of drama3-small.", first.codes)], sampling=Sampling(seed=8))
    print(f"kept voice: {len(kept.audio) / kept.sample_rate:.1f}s audio in {time.time() - t:.1f}s {kept.chunks}")
    write_wav(out / "kept.wav", kept.audio, kept.sample_rate)
    print(f"wrote {out}/generated.wav and {out}/kept.wav")


if __name__ == "__main__":
    main()
