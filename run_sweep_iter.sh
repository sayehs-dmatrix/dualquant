#!/bin/bash
# Sweep identical to run_sweep.sh but includes dualquant num_iter in the
# results CSV filename so runs with different iteration counts don't collide.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_TRUST_REMOTE_CODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

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

# ── Methods ─────────────────────────────────────────────────────────────────────
METHODS=(
    # rtn
    dualquant
    # gptq_seq
    # gptq
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
    # rtn_int8
)

# ── Per-format canonical (block_size, scale_format) ────────────────────────────
fmt_defaults() {
    case "$1" in
        mxfp4|mxint4|mxfp8_e4m3|mxfp8_e5m2|mxint8) echo "32 e8m0" ;;
        nvfp4)                                     echo "16 e4m3" ;;
        sfp4)                                      echo "16 e4m4" ;;
        rtn_int4)                                  echo "64 none" ;;
        rtn_int8)                                  echo "64 none" ;;
        *) echo "unknown format $1" >&2; exit 1 ;;
    esac
}

# ── Activation quantisation ────────────────────────────────────────────────────
ACT_QUANT="${ACT_QUANT:-on}"
ACT_FMT="${ACT_FMT:-rtn_int4}"
ACT_BLOCK_SIZE="${ACT_BLOCK_SIZE:-64}"
ACT_SCALE_FORMAT="${ACT_SCALE_FORMAT:-none}"

# ── Preprocess ─────────────────────────────────────────────────────────────────
PREPROCESS="${PREPROCESS:-smoothquant}"

sq_scales_for() {
    case "$1" in
        "meta-llama/Llama-3.1-8B")   echo "smoothquant/act_scales/Llama-3.1-8B_seq_len_8192_4bit.pt" ;;
        "meta-llama/Llama-3.2-1B")   echo "smoothquant/act_scales/Llama-3.2-1B_seq_len_8192_4bit.pt" ;;
        "meta-llama/Llama-2-7b-hf")  echo "smoothquant/act_scales/Meta-Llama-2-7B_seqlen_4096_4bit.pt" ;;
        "meta-llama/Llama-2-13b-hf") echo "smoothquant/act_scales/Meta-Llama-2-13B_seqlen_4096_4bit.pt" ;;
        "mistralai/Mistral-7B-v0.3") echo "smoothquant/act_scales/Mistral-7B-v01_seq_len_8192_4bit.pt" ;;
        "Qwen/Qwen2-1.5B")           echo "smoothquant/act_scales/Qwen2-1.5B_seq_len_8192_4bit.pt" ;;
        "Qwen/Qwen2.5-0.5B")         echo "smoothquant/act_scales/Qwen2.5-0.5B_seq_len_8192_4bit.pt" ;;
        "Qwen/Qwen2.5-7B")           echo "smoothquant/act_scales/Qwen2.5-7B_seq_len_8192_4bit.pt" ;;
        "Qwen/Qwen2.5-14B")          echo "smoothquant/act_scales/Qwen2.5-14B_seq_len_8192_4bit.pt" ;;
        "Qwen/Qwen3-0.6B")           echo "smoothquant/act_scales/Qwen3-0.6B_seq_len_8192_4bit.pt" ;;
        "Qwen/Qwen3-1.7B")           echo "smoothquant/act_scales/Qwen3-1.7B_seq_len_8192_4bit.pt" ;;
        "Qwen/Qwen3-14B")            echo "smoothquant/act_scales/Qwen3-14B_seq_len_8192_4bit.pt" ;;
        *) echo ""; return 1 ;;
    esac
}

# ── Calibration / eval ─────────────────────────────────────────────────────────
NSAMPLES=128
SEQLEN=2048
TASKS=wikitext

# ── Method hyperparams JSON ────────────────────────────────────────────────────
METHOD_CFG="${SCRIPT_DIR}/configs/methods.json"

# Read num_iter from the config — included in the CSV filename for dualquant runs.
DQ_ITER=$(python3 -c "import json; print(json.load(open('${METHOD_CFG}'))['dualquant']['num_iter'])")

# ── Sweep ──────────────────────────────────────────────────────────────────────
TOTAL=$(( ${#MODELS[@]} * ${#METHODS[@]} * ${#WEIGHT_FMTS[@]} ))
COUNT=0

for model in "${MODELS[@]}"; do
    model_tag=$(echo "${model}" | tr '/' '_')

    for method in "${METHODS[@]}"; do
        # Include num_iter in filename for dualquant so runs with different
        # iteration counts produce separate CSVs.
        if [[ "${method}" == "dualquant" ]]; then
            RESULTS_CSV="${RESULTS_DIR}/${model_tag}_iter${DQ_ITER}.csv"
        else
            RESULTS_CSV="${RESULTS_DIR}/${model_tag}.csv"
        fi

        for fmt in "${WEIGHT_FMTS[@]}"; do
            COUNT=$(( COUNT + 1 ))
            read -r BLOCK_SIZE SCALE_FORMAT <<< "$(fmt_defaults "${fmt}")"

            echo ""
            echo "═══════════════════════════════════════════════════════════"
            echo " Run ${COUNT}/${TOTAL}: model=${model}"
            echo "                       method=${method}  iter=${DQ_ITER}  weight=${fmt}/${BLOCK_SIZE}/${SCALE_FORMAT}"
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
                    echo "  [SKIP] no smoothquant scales mapping for ${model}"
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
