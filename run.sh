#!/bin/bash
# One-shot: install -> demo videos -> tests -> server.
set -e
cd "$(dirname "$0")"
if command -v conda >/dev/null 2>&1 && [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
	eval "$(conda shell.bash hook)"
	conda activate "${VSD_CONDA_ENV:-env_p13}"
fi
python -m pip install -q -r requirements.txt
python scripts/make_demo_videos.py
python -m pytest tests/ -x -q
echo "--- starting server on :8000 ---"
exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
