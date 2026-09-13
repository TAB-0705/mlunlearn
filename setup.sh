#!/usr/bin/env bash
# Environment setup for the Machine Unlearning Verification project.
#
#   bash setup.sh
#
# Creates a local virtual environment and installs dependencies. On your RTX 3050
# laptop, install the CUDA build of torch so training uses the GPU:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
# (match cu121 to your installed CUDA/driver version). The CPU build also works,
# just slower — Purchase-100 is small enough that CPU is fine for early dev.

set -e

python3 -m venv .venv
# activate: Linux/macOS ->  source .venv/bin/activate
#          Windows PowerShell -> .venv\Scripts\Activate.ps1
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo ""
echo "Environment ready. Verify with:"
echo "  .venv/bin/python -m tests.test_spine      # foundation"
echo "  .venv/bin/python -m tests.test_training   # models train + membership gap"
echo "  .venv/bin/python -m core.build_anchors    # end-to-end demo (synthetic)"
