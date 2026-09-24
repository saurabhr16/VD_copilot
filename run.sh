#!/bin/bash
# One-shot: install -> demo videos -> tests -> server.
set -e
cd "$(dirname "$0")"
pip install -q -r requirements.txt
python scripts/make_demo_videos.py
python -m pytest tests/ -x -q
echo "--- starting server on :8000 ---"
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
