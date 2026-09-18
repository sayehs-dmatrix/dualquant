#!/bin/bash
# Per-layer SQNR sweep: FP vs DualQuant vs SmoothQuant.
# Same MODELS × WEIGHT_FMTS axes as run_sweep.sh.
# Results written to results/sqnr/<model_tag>_<fmt>.csv
#
# Override via env vars:
#   ACT_QUANT=off NSAMPLES=8 bash run_sqnr_per_layer.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_TRUST_REMOTE_CODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

RESULTS_DIR="results/sqnr_isolate_layer_0"
mkdir -p "${RESULTS_DIR}"

# ── Models ─────────────────────────────────────────────────────────────────────
MODELS=(
    # "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.1-8B"
    # "meta-llama/Llama-2-7b-hf"
    # "Qwen/Qwen3-0.6B"
    # "Qwen/Qwen3-14B"
    # "Qwen/Qwen2.5-7B"
    # "mistralai/Mistral-7B-v0.3"
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
        nvfp4)                                        echo "16 e4m3" ;;
        sfp4)                                         echo "16 e4m4" ;;
        rtn_int4)                                     echo "64 none" ;;
        rtn_int8)                                     echo "64 none" ;;
        *) echo "unknown format $1" >&2; exit 1 ;;
    esac
}

# ── Activation quantisation (off by default; flip ACT_QUANT=on to enable) ──────
# ACT_QUANT="${ACT_QUANT:-off}"
# ACT_FMT="${ACT_FMT:-rtn_int4}"
# ACT_BLOCK_SIZE="${ACT_BLOCK_SIZE:-64}"
# ACT_SCALE_FORMAT="${ACT_SCALE_FORMAT:-none}"

ACT_QUANT="${ACT_QUANT:-off}"
ACT_FMT="${ACT_FMT:-mxfp4}"
ACT_BLOCK_SIZE="${ACT_BLOCK_SIZE:-32}"
ACT_SCALE_FORMAT="${ACT_SCALE_FORMAT:-e8m0}"

# ── Sub-layer selection ────────────────────────────────────────────────────────
ATTN_LAYERS="${ATTN_LAYERS:-on}"   # on | off  (--no-attn-layers to skip)
MLP_LAYERS="${MLP_LAYERS:-on}"     # on | off  (--no-mlp-layers to skip)

# ── Single-block isolation (unset = quantise all layers) ───────────────────────
# Set to a layer index to quantise ONLY that block; all others restored to FP.
# Example: ISOLATE_LAYER=1 bash run_sqnr_per_layer.sh
ISOLATE_LAYER="${ISOLATE_LAYER:-0}"

# ── Data split ─────────────────────────────────────────────────────────────────
# test       (default): WikiText-2 test split, chunked into seqlen tokens
# test_indiv          : WikiText-2 test split, individual text tokenisation
#                       (same tokenisation as sq_calib — fair train vs test comparison)
# sq_calib            : WikiText-2 train split, shuffle seed=42, individual tokenisation
#                       (exact same texts as SmoothQuant calibration)
DATA_SPLIT="${DATA_SPLIT:-test}"

# ── Calibration ────────────────────────────────────────────────────────────────
NSAMPLES="${NSAMPLES:-16}"
# seqlen=2048 for test (chunked) mode — gate/up_proj outputs are (seqlen × 8192),
# so 8192 tokens causes ~14 GB CPU RAM per sample → OOM kills SSH.
# sq_calib/test_indiv use individual texts (often <<8192 tokens) so 8192 is safe there.
SEQLEN="${SEQLEN:-4096}"

# ── Method hyperparams JSON ────────────────────────────────────────────────────
METHOD_CFG="configs/methods.json"

# ── Sweep ──────────────────────────────────────────────────────────────────────
TOTAL=$(( ${#MODELS[@]} * ${#WEIGHT_FMTS[@]} ))
COUNT=0

for model in "${MODELS[@]}"; do
    model_tag=$(echo "${model}" | tr '/' '_')

    for fmt in "${WEIGHT_FMTS[@]}"; do
        COUNT=$(( COUNT + 1 ))
        read -r BLOCK_SIZE SCALE_FORMAT <<< "$(fmt_defaults "${fmt}")"

        # Build filename suffix to distinguish runs
        SUFFIX="_act${ACT_QUANT}"
        [[ -n "${ISOLATE_LAYER}" ]]           && SUFFIX="${SUFFIX}_isolate${ISOLATE_LAYER}"
        [[ "${DATA_SPLIT}" != "test" ]]       && SUFFIX="${SUFFIX}_${DATA_SPLIT}"
        CSV_OUT="${RESULTS_DIR}/${model_tag}_${fmt}${SUFFIX}.csv"

        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo " Run ${COUNT}/${TOTAL}: model=${model}"
        echo "                       weight=${fmt}/${BLOCK_SIZE}/${SCALE_FORMAT}"
        echo "                       act_quant=${ACT_QUANT}  act_fmt=${ACT_FMT}"
        echo "═══════════════════════════════════════════════════════════"

        ARGS=(
            --model               "${model}"
            --weight-fmt          "${fmt}"
            --weight-block-size   "${BLOCK_SIZE}"
            --weight-scale-format "${SCALE_FORMAT}"
            --nsamples            "${NSAMPLES}"
            --seqlen              "${SEQLEN}"
            --data-split          "${DATA_SPLIT}"
            --method-cfg          "${METHOD_CFG}"
            --csv-out             "${CSV_OUT}"
        )

        if [[ "${ACT_QUANT}" == "on" ]]; then
            ARGS+=(
                --act-quant              on
                --act-fmt                "${ACT_FMT}"
                --act-block-size         "${ACT_BLOCK_SIZE}"
                --act-scale-format       "${ACT_SCALE_FORMAT}"
                --act-scaled-before-quant
            )
        fi

        if [[ "${ATTN_LAYERS}" == "off" ]]; then ARGS+=(--no-attn-layers); fi
        if [[ "${MLP_LAYERS}"  == "off" ]]; then ARGS+=(--no-mlp-layers);  fi
        if [[ -n "${ISOLATE_LAYER}" ]]; then ARGS+=(--isolate-layer "${ISOLATE_LAYER}"); fi

        python -u ablation/sqnr_per_layer.py "${ARGS[@]}" \
            || echo "  [FAILED] ${model} / ${fmt} — skipping"
    done
done

echo ""
echo "Sweep complete. Results in: ${RESULTS_DIR}/"
