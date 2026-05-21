# Evaluator code from GPTAQ
import torch
import os
from tqdm import tqdm
# from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
# from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def evaluator_newVersion(model, tokenizer, dataset_enc, dev, arg, dataset):
    # breakpoint()
    # max_length=2048
    # stride=512
    device="cuda"

    # breakpoint()
    if hasattr(model.config, "max_position_embeddings"):
        max_seq_len = model.config.max_position_embeddings
    elif hasattr(model.config, "n_positions"):
        max_seq_len = model.config.n_positions
    else:
        max_seq_len = 4096
    # breakpoint()
    ###### Cap at 8192 but never exceed the model's trained RoPE range
    ###### (Llama-2: 4096; Llama-3: 131072; Qwen2/3: 32768+).
    max_seq_len = min(8192, model.config.max_position_embeddings)
    stride = max_seq_len
    # breakpoint()
    model.eval()
    model.to(device)

    

    # Tokenize the entire dataset
    # encodings = tokenizer("\n\n".join(dataset['text']), return_tensors="pt")
    if dataset=='test':
        encodings = dataset_enc
        seq_len = encodings.input_ids.size(1)
    if dataset=='train':
        encodings = dataset_enc
        seq_len = encodings.input_ids.size(1)//10
    
    # encodings = input_ids

    nlls = []  # Store negative log likelihoods
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_seq_len, seq_len)
        trg_len = end_loc - prev_end_loc  # May be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        # Mask out the loss calculation on the context part
        target_ids[:, :-trg_len] = -100
        
        
        with torch.no_grad():
            # Standard forward pass with labels
            outputs = model(input_ids, labels=target_ids)
            # Loss is the mean negative log likelihood per token.
            # Multiply by trg_len to get the sum for the new tokens in this window.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood.to(model.device))

    
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Calculate final perplexity
    # ppl = torch.exp(torch.stack(nlls).sum() / end_loc) # end_loc is the total length

    loss = torch.stack(nlls).float().sum() / end_loc
    ppl = torch.exp(loss)

    return ppl.item()

# Usage Example:
# ppl = evaluate_perplexity(model, tokenizer, wikitext2_test_text)
# print(f"Perplexity: {ppl:.2f}")
