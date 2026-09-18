"""WikiText-2 perplexity through the REAL packed W4A4 kernel, and for the BF16 baseline,
under identical settings so the two are directly comparable.

Why not the repo's legacy evaluator: it sets its window to
model.config.max_position_embeddings, which is 131072 for Llama-3.1-8B, and then asks
HuggingFace to compute the loss over the whole window. That materialises fp32 logits of
[window, 128256] and OOMs on a 24GB GPU at ANY precision (BF16 included), so it cannot
produce a comparable pair of numbers here.

This uses the standard convention reported by quantization papers (GPTQ/AWQ-style):
concatenate the test split, cut it into non-overlapping windows of `--seqlen` tokens, and
average the next-token NLL over all windows. Cross-entropy is accumulated in fp32 over
slices of the sequence so peak memory stays small.

Usage:
    python ppl_eval.py --mode bf16
    python ppl_eval.py --mode int4
"""
import os
import argparse, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_ROOT, _HERE,
          os.environ.get("NUNCHAKU_DIR", "")):
    if p and p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
import transformers
from main import load_model, build_cfg, _load_all_hyperparams
import datautils


@torch.no_grad()
def ppl(model, ids, seqlen, dev, chunk=256):
    n = ids.numel() // seqlen
    total_nll, total_tok = 0.0, 0
    for i in range(n):
        w = ids[:, i * seqlen:(i + 1) * seqlen].to(dev)
        logits = model(w).logits            # [1, seqlen, V]
        # shift: predict token t+1 from position t
        src = logits[:, :-1, :].squeeze(0)
        tgt = w[:, 1:].squeeze(0)
        for s in range(0, src.shape[0], chunk):
            sl = src[s:s + chunk].float()
            total_nll += F.cross_entropy(sl, tgt[s:s + chunk], reduction="sum").item()
        total_tok += tgt.numel()
        del logits, src
        if (i + 1) % 20 == 0:
            print(f"    window {i+1}/{n}  running PPL={torch.tensor(total_nll/total_tok).exp():.4f}",
                  flush=True)
    return float(torch.tensor(total_nll / total_tok).exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--mode", choices=["bf16", "int4"], required=True)
    ap.add_argument("--seqlen", type=int, default=2048,
                    help="Use 8192 to match the paper's reported PPL setting; 2048 is the\n                          standard GPTQ-style convention. PPL is not comparable across these.")
    cli = ap.parse_args()
    dev = torch.device("cuda:0")

    if cli.mode == "int4":
        # forward_beta: correct on the default stream (no CUDA-graph capture here, so no
        # custom-stream requirement). This is the same packed W4A4 path used for the
        # throughput numbers.
        os.environ["NUNCHAKU_FWD_MODE"] = "nocache"
        import full_model_nunchaku as fmn
        fmn.NunchakuW4A4Linear._FWD_MODE = "nocache"
        args = fmn._make_args(cli.model, 128, cli.seqlen)
        cfg = build_cfg(args, _load_all_hyperparams(args.method_cfg))
        cfg["_model_id"] = cli.model
        model = load_model(cli.model)
        fmn.convert_model_to_nunchaku(model, cfg)
        model = model.to(dev).eval()
    else:
        model = load_model(cli.model).to(dev).eval()
    model.config.use_cache = False

    _, _, _, testenc, _ = datautils.get_loaders(
        "wikitext", cli.model, nsamples=128, seed=0, seqlen=cli.seqlen)
    ids = testenc.input_ids if hasattr(testenc, "input_ids") else testenc
    print(f"{cli.mode}: test tokens={ids.numel()}  windows={ids.numel()//cli.seqlen} "
          f"seqlen={cli.seqlen}", flush=True)

    val = ppl(model, ids, cli.seqlen, dev)
    print(f"\nRESULT {cli.mode} WikiText-2 PPL (seqlen={cli.seqlen}, real packed path"
          f"{' ' if cli.mode=='int4' else ' n/a'}): {val:.4f}")


if __name__ == "__main__":
    main()
