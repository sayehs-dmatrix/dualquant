#!/bin/bash
# Run SmoothQuant alpha sweep model by model.
# Each model runs all 4 scenarios × 11 alphas = 44 jobs.
# Comment out models you don't want to run.
#
# ── Scenario index reference ──────────────────────────────────────────────────
#   0  SQ + INT4  W4A16   weights INT4  (group-64),  activations FP16
#   1  SQ + INT4  W4A4    weights INT4  (group-64),  activations INT4  (group-64)
#   2  SQ + MXFP4 W4A16   weights MXFP4 (block-32),  activations FP16
#   3  SQ + MXFP4 W4A4    weights MXFP4 (block-32),  activations MXINT4 (block-32)
#
# ── Usage examples ────────────────────────────────────────────────────────────
#   bash run_sq_alpha_sweep.sh                             # all models, all scenarios
#   MODEL="meta-llama/Llama-3.1-8B" bash run_sq_alpha_sweep.sh          # one model
#   MODEL="meta-llama/Llama-3.1-8B" SCENARIOS="1 3" bash run_sq_alpha_sweep.sh  # A4W4 only
#   SCENARIOS="0 2" bash run_sq_alpha_sweep.sh             # W4A16 only (2× faster)
#   ALPHAS="0.3 0.5 0.7" bash run_sq_alpha_sweep.sh        # custom alpha grid

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONBREAKPOINT=0

SWEEP="python sweep_sq_alpha_quant.py"

# Optional overrides via env
ALPHAS="${ALPHAS:-}"            # default (unset) = 0.0 0.1 0.2 ... 1.0 (11 values); override e.g. ALPHAS="0.3 0.5 0.7"
SCENARIOS="${SCENARIOS:-2 3}"      # e.g. SCENARIOS="0 1"
# MODEL="meta-llama/Llama-3.1-8B" # MODEL=""  # default (empty) = run all models; override e.g. MODEL="meta-llama/Llama-3.1-8B"

extra_args=()
[[ -n "${ALPHAS}"    ]] && extra_args+=(--alphas    ${ALPHAS})
[[ -n "${SCENARIOS}" ]] && extra_args+=(--scenarios ${SCENARIOS})

# ── If MODEL env var is set, run just that one model ──────────────────────────
if [[ -n "${MODEL:-}" ]]; then
    echo "Running single model: ${MODEL}"
    ${SWEEP} --models "${MODEL}" "${extra_args[@]}"
    exit 0
fi

# ── Otherwise run each model in sequence ──────────────────────────────────────
run_model() {
    local model="$1"
    echo ""
    echo "████████████████████████████████████████████████████████████"
    echo "  MODEL: ${model}"
    echo "████████████████████████████████████████████████████████████"
    ${SWEEP} --models "${model}" "${extra_args[@]}" \
        || echo "  [FAILED] ${model} — continuing to next model"
    echo ""
}

# Comment out any model you want to skip
# run_model "Qwen/Qwen3-0.6B"
# run_model "meta-llama/Llama-3.2-1B"
# run_model "meta-llama/Llama-2-7b-hf"
# run_model "Qwen/Qwen2.5-7B"
# run_model "meta-llama/Llama-3.1-8B"
run_model "Qwen/Qwen3-14B"

echo "All models done. Results in: results_sq_alpha_sweep/sq_alpha_quant_sweep.csv"
