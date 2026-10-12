#!/bin/zsh
# 로컬 서버 실행: ./run.sh  (http://127.0.0.1:8765)
cd "$(dirname "$0")"
mkdir -p logs
exec uv run uvicorn app.main:app --host 127.0.0.1 --port 8765 >> logs/server.log 2>&1
