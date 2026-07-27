#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

python_command="${PYTHON_BIN:-python3}"
venv_dir="${TYPHOON_VENV_DIR:-.venv}"
model_path="typhoon/models/soft_moe_vd_warmstart_ppo/seed-42/service_candidate.pt"

if [[ ! -f "${model_path}" ]]; then
  echo "Missing tracked Phase 14 checkpoint: ${model_path}" >&2
  exit 1
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  "${python_command}" -m venv "${venv_dir}"
fi

"${venv_dir}/bin/python" -m pip install -e .

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TYPHOON_HOST="${TYPHOON_HOST:-0.0.0.0}"
export TYPHOON_PORT="${TYPHOON_PORT:-8765}"

echo "Phase 14 API: http://127.0.0.1:${TYPHOON_PORT}"
echo "OpenAPI docs: http://127.0.0.1:${TYPHOON_PORT}/docs"
echo "Stop with Ctrl-C."
exec "${venv_dir}/bin/python" -m typhoon.api
