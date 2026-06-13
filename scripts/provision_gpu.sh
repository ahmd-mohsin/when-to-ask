#!/usr/bin/env bash
# Provision a rented GPU box (RunPod / Lambda) for a real de-risk run.
# The dev machine (AMD iGPU, no CUDA) can only run the CPU smoke path; the white-box
# backbone (hidden states) and the Llama-3.3-70B judge need a real GPU.
#
# Recommended instance: 1x H100 80GB or 1x A100 80GB, judge via API (see below).
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA="${CUDA:-cu121}"          # match the box's CUDA toolkit
BACKBONE="${BACKBONE:-Qwen/Qwen2.5-Coder-7B-Instruct}"

echo "== 1/4 base + project (editable) =="
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

echo "== 2/4 GPU deps =="
python -m pip install "torch>=2.2" --index-url "https://download.pytorch.org/whl/${CUDA}"
python -m pip install -r requirements-gpu.txt

echo "== 3/4 vendored repos =="
bash scripts/clone_third_party.sh || pwsh scripts/clone_third_party.ps1 || true

echo "== 4/4 download backbone + verify a hidden-state forward pass =="
python - "$BACKBONE" <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = sys.argv[1]
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="auto",
                                             output_hidden_states=True)
ids = tok("def add(a, b):", return_tensors="pt").to(model.device)
out = model(**ids)
print("OK hidden_states:", len(out.hidden_states), "x", tuple(out.hidden_states[len(out.hidden_states)//2].shape))
PY

cat <<'NOTE'

Backbone ready. Now point the judge at a hosted Llama-3.3-70B-Instruct endpoint:

  export JUDGE_BASE_URL="https://api.together.xyz/v1"   # or Fireworks / DeepInfra
  export JUDGE_API_KEY="..."

  # Self-host alternative (needs 2x A100-80GB or 1x H100 + offload):
  #   vllm serve casperhansen/llama-3.3-70b-instruct-awq --port 8000
  #   export JUDGE_BASE_URL="http://localhost:8000/v1"  JUDGE_API_KEY="EMPTY"

Then run:  python scripts/run_derisk.py --config configs/derisk.yaml
NOTE
