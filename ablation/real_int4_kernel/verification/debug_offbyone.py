import os, sys
sys.path.insert(0, _os.path.join(_REPO_ROOT, "ablation/real_int4_kernel"))
sys.path.insert(0, _REPO_ROOT)
import torch, transformers
from transformers import StaticCache
from main import load_model

import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root from this file

dev = torch.device("cuda:0")
model_id = "meta-llama/Llama-3.1-8B"
model = load_model(model_id)
model.config.use_cache = True
model = model.to(dev).eval()
tok_ = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=False)

prompt = "The quick brown fox jumps over the lazy dog. " * 8
ids = tok_(prompt, return_tensors="pt").input_ids.to(dev)
print("prompt last 5 ids:", ids[0, -5:].tolist())

with torch.no_grad():
    # ---- eager ----
    out = model(ids, use_cache=True)
    past = out.past_key_values
    T1_eager = out.logits[:, -1:].argmax(-1)
    print("EAGER T1 (from prefill):", T1_eager.item(), repr(tok_.decode(T1_eager[0])))
    out2 = model(T1_eager, past_key_values=past, use_cache=True)
    T2_eager = out2.logits[:, -1:].argmax(-1)
    print("EAGER T2 (feed T1):     ", T2_eager.item(), repr(tok_.decode(T2_eager[0])))

    # ---- StaticCache ----
    cache = StaticCache(config=model.config, batch_size=1, max_cache_len=ids.shape[1] + 64,
                        device=dev, dtype=torch.bfloat16)
    pos = torch.arange(ids.shape[1], device=dev)
    out = model(ids, past_key_values=cache, cache_position=pos, use_cache=True)
    T1_sc = out.logits[:, -1:].argmax(-1)
    print("STATIC T1 (from prefill):", T1_sc.item(), repr(tok_.decode(T1_sc[0])))

    static_input = T1_sc.clone()
    static_pos = torch.tensor([ids.shape[1]], device=dev, dtype=torch.long)
    lg = model(input_ids=static_input, past_key_values=cache,
               cache_position=static_pos, use_cache=True).logits
    T2_sc = lg[:, -1:].argmax(-1)
    print("STATIC T2 (feed T1):     ", T2_sc.item(), repr(tok_.decode(T2_sc[0])))
    print()
    print("T1 match:", T1_eager.item() == T1_sc.item(), " T2 match:", T2_eager.item() == T2_sc.item())
    print("max |logit diff| on first decode step:",
          (out2.logits[:, -1, :].float() - lg[:, -1, :].float()).abs().max().item())
