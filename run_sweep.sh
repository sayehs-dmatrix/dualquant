#!/bin/bash
# Sweep over MODELS × METHODS × WEIGHT_FMTS.
# Results are appended to a per-model CSV under results/.
#
# Comment out items in the arrays to skip them. Each weight format picks its
# canonical block_size + scale_format via fmt_defaults() below.

set -euo pipefail

# ── GPU / env ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_TRUST_REMOTE_CODE=1
# Needs an nvcc that supports c++20 (CUDA >= 12.x). Override CUDA_HOME for your
# machine; the default below is only used if it actually exists, so an unset or
# wrong value fails loudly at nvcc time rather than silently pointing nowhere.
: "${CUDA_HOME:=<HOME>}"
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "warning: no nvcc at ${CUDA_HOME}/bin/nvcc -- set CUDA_HOME to a CUDA >= 12.x install" >&2
fi
export CUDA_HOME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Output directory ───────────────────────────────────────────────────────────
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# ── Models ─────────────────────────────────────────────────────────────────────
MODELS=(
    # "Qwen/Qwen3-0.6B"
    # "Qwen/Qwen2.5-7B"
    # "Qwen/Qwen3-14B"
    # "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.2-1B"
    # "meta-llama/Llama-2-7b-hf"
    # "mistralai/Mistral-7B-v0.3"
    # "google/gemma-2-9b"
)

# ── Methods (one row of the new --method flag) ─────────────────────────────────
METHODS=(
    # rtn
    # dualquant
    gptq_seq       # sequential, slower, required for rtn_int4/int8
    # gptq          # parallel, fast, works for block formats (mxfp4, mxint4, etc.)     
    # awq
    # sinq
)

# ── Weight formats ─────────────────────────────────────────────────────────────
WEIGHT_FMTS=(
    # mxfp4
    # mxint4
    # mxfp8_e4m3
    # mxfp8_e5m2
    # mxint8
    # nvfp4
    # sfp4
    rtn_int4
    # rtn_int4_asym
    # rtn_int8
)

# ── Per-format canonical (block_size, scale_format) ────────────────────────────
fmt_defaults() {
    case "$1" in
        mxfp4|mxint4|mxfp8_e4m3|mxfp8_e5m2|mxint8) echo "32 e8m0" ;;
        nvfp4)                                     echo "16 e4m3" ;;
        sfp4)                                      echo "16 e4m4" ;;
        rtn_int4)                                  echo "64 none" ;;
        rtn_int4_asym)                             echo "64 none" ;;
        rtn_int8)                                  echo "64 none" ;;   # per-channel (block_size unused)
        *) echo "unknown format $1" >&2; exit 1 ;;
    esac
}

# ── Activation quantisation (off by default; flip ACT_QUANT=on to enable) ──────
ACT_QUANT="${ACT_QUANT:-off}"
ACT_FMT="${ACT_FMT:-rtn_int4}"
ACT_BLOCK_SIZE="${ACT_BLOCK_SIZE:-64}"
ACT_SCALE_FORMAT="${ACT_SCALE_FORMAT:-none}"

# ACT_QUANT="${ACT_QUANT:-on}"
# ACT_FMT="${ACT_FMT:-mxint4}"
# ACT_BLOCK_SIZE="${ACT_BLOCK_SIZE:-32}"
# ACT_SCALE_FORMAT="${ACT_SCALE_FORMAT:-e8m0}"

# ACT_QUANT="${ACT_QUANT:-on}"
# ACT_FMT="${ACT_FMT:-mxint4}"
# ACT_BLOCK_SIZE="${ACT_BLOCK_SIZE:-32}"
# ACT_SCALE_FORMAT="${ACT_SCALE_FORMAT:-e8m0}"

# ── Preprocess (none | smoothquant | quarot) ──────────────────────────────────
#   Override via env: PREPROCESS=quarot bash run_sweep.sh
#   QuaRot always applies R1+R2+R4 regardless of ACT_QUANT (fixed 2026-05-27).
PREPROCESS="${PREPROCESS:-none}"

# ── SmoothQuant per-model scales (only consulted when PREPROCESS=smoothquant) ──
# Maps each model in MODELS=(...) to its specific .pt file under
# smoothquant/act_scales/. Add a case for any new model you sweep.
sq_scales_for() {
    case "$1" in
        "meta-llama/Llama-3.1-8B")        echo "smoothquant/act_scales/Llama-3.1-8B_seq_len_8192_4bit.pt" ;;
        "meta-llama/Llama-3.2-1B")        echo "smoothquant/act_scales/Llama-3.2-1B_seq_len_8192_4bit.pt" ;;
        "meta-llama/Llama-2-7b-hf")       echo "smoothquant/act_scales/Meta-Llama-2-7B_seqlen_4096_4bit.pt" ;;
        "meta-llama/Llama-2-13b-hf")      echo "smoothquant/act_scales/Meta-Llama-2-13B_seqlen_4096_4bit.pt" ;;
        "mistralai/Mistral-7B-v0.3")      echo "smoothquant/act_scales/Mistral-7B-v01_seq_len_8192_4bit.pt" ;;
        "Qwen/Qwen2-1.5B")                echo "smoothquant/act_scales/Qwen2-1.5B_seq_len_8192_4bit.pt" ;;
        "Qwen/Qwen2.5-0.5B")              echo "smoothquant/act_scales/Qwen2.5-0.5B_seq_len_8192_4bit.pt" ;;
        "Qwen/Qwen2.5-7B")                echo "smoothquant/act_scales/Qwen2.5-7B_seq_len_8192_4bit.pt" ;;
        "Qwen/Qwen2.5-14B")               echo "smoothquant/act_scales/Qwen2.5-14B_seq_len_8192_4bit.pt" ;;
        "Qwen/Qwen3-0.6B")                echo "smoothquant/act_scales/Qwen3-0.6B_seq_len_8192_4bit.pt" ;;
        "Qwen/Qwen3-1.7B")                echo "smoothquant/act_scales/Qwen3-1.7B_seq_len_8192_4bit.pt" ;;
        "Qwen/Qwen3-14B")                 echo "smoothquant/act_scales/Qwen3-14B_seq_len_8192_4bit.pt" ;;
        *) echo ""; return 1 ;;
    esac
}

# ── Calibration / eval ─────────────────────────────────────────────────────────
NSAMPLES=128
SEQLEN=2048
TASKS=wikitext     # 'wikitext' | 'c4' | 'wikitext c4'

# ── Method hyperparams JSON (single merged file) ───────────────────────────────
METHOD_CFG="${SCRIPT_DIR}/configs/methods.json"

# ── Sweep ──────────────────────────────────────────────────────────────────────
TOTAL=$(( ${#MODELS[@]} * ${#METHODS[@]} * ${#WEIGHT_FMTS[@]} ))
COUNT=0

for model in "${MODELS[@]}"; do
    model_tag=$(echo "${model}" | tr '/' '_')
    RESULTS_CSV="${RESULTS_DIR}/${model_tag}.csv"

    for method in "${METHODS[@]}"; do
        for fmt in "${WEIGHT_FMTS[@]}"; do
            COUNT=$(( COUNT + 1 ))
            read -r BLOCK_SIZE SCALE_FORMAT <<< "$(fmt_defaults "${fmt}")"

            echo ""
            echo "═══════════════════════════════════════════════════════════"
            echo " Run ${COUNT}/${TOTAL}: model=${model}"
            echo "                       method=${method}  weight=${fmt}/${BLOCK_SIZE}/${SCALE_FORMAT}"
            echo "                       act_quant=${ACT_QUANT}  preprocess=${PREPROCESS}"
            echo "═══════════════════════════════════════════════════════════"

            ARGS=(
                --model                 "${model}"
                --method                "${method}"
                --method-cfg            "${METHOD_CFG}"
                --weight-fmt            "${fmt}"
                --weight-block-size     "${BLOCK_SIZE}"
                --weight-scale-format   "${SCALE_FORMAT}"
                --tasks                 ${TASKS}
                --nsamples              "${NSAMPLES}"
                --seqlen                "${SEQLEN}"
                --results-path          "${RESULTS_CSV}"
                --preprocess            "${PREPROCESS}"
            )
            # MSE-optimal weight clipping for rtn_int4/rtn_int8 + QuaRot
            # (required with QuaRot rotations; ignored silently for other formats)
            if [[ "${fmt}" == rtn_int* && "${PREPROCESS}" == "quarot" ]]; then
                ARGS+=(--weight-clip)
            fi
            if [[ "${ACT_QUANT}" == "on" ]]; then
                ARGS+=(
                    --act-quant            on
                    --act-fmt              "${ACT_FMT}"
                    --act-block-size       "${ACT_BLOCK_SIZE}"
                    --act-scale-format     "${ACT_SCALE_FORMAT}"
                    --act-scaled-before-quant
                )
            fi
            if [[ "${PREPROCESS}" == "smoothquant" ]]; then
                SQ_PATH="$(sq_scales_for "${model}")"
                if [[ -z "${SQ_PATH}" ]]; then
                    echo "  [SKIP] no smoothquant scales mapping for ${model} (add to sq_scales_for in run_sweep.sh)"
                    continue
                fi
                ARGS+=(--sq-scales "${SQ_PATH}")
            fi

            python main.py "${ARGS[@]}" \
                || echo "  [FAILED] ${model} / ${method} / ${fmt} — skipping"
        done
    done
done

echo ""
echo "Sweep complete. Results in: ${RESULTS_DIR}/"
