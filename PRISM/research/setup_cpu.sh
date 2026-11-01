#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_dir="$repo_root/.venv"
requirements="$repo_root/PRISM/research/requirements.txt"
encoder_requirements="$repo_root/PRISM/research/requirements-encoders.txt"
archive="$repo_root/PRISM/research/artifacts/full_bge_cpu.tar.gz"
checksum="$archive.sha256"
artifact_root="$(dirname "$archive")"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin="$PYTHON_BIN"
elif command -v python3.10 >/dev/null 2>&1; then
    python_bin="$(command -v python3.10)"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
else
    printf 'Python 3.10 or newer is required.\n' >&2
    exit 1
fi

"$python_bin" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' || {
    printf 'Found %s; this setup requires Python >=3.10 and <3.14.\n' "$($python_bin --version 2>&1)" >&2
    exit 1
}

if [[ ! -x "$venv_dir/bin/python" ]]; then
    if [[ -e "$venv_dir" ]]; then
        printf '%s exists but is not a usable Linux/macOS virtual environment. Move it aside before setup.\n' "$venv_dir" >&2
        exit 1
    fi
    "$python_bin" -m venv "$venv_dir" || {
        printf 'Could not create venv. On Debian/Ubuntu, install the matching python%s-venv OS package, then rerun.\n' "$($python_bin -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" >&2
        exit 1
    }
fi

venv_python="$venv_dir/bin/python"
if ! "$venv_python" -c 'import numpy, pandas, sklearn, torch, sentence_transformers, shap, matplotlib' >/dev/null 2>&1; then
    "$venv_python" -m pip install --upgrade pip
    "$venv_python" -m pip install -r "$requirements"
    "$venv_python" -m pip install -r "$encoder_requirements"
fi

"$venv_python" -m pip check
CUDA_VISIBLE_DEVICES="" "$venv_python" -c 'import torch; print(f"PyTorch {torch.__version__}; inference device: CPU; CUDA hidden")'

if [[ -f "$archive" && -f "$checksum" ]]; then
    if head -n 1 "$archive" | grep -q '^version https://git-lfs.github.com/spec/v1$'; then
        if command -v git-lfs >/dev/null 2>&1 && git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            git -C "$repo_root" lfs pull --include='PRISM/research/artifacts/full_bge_cpu.tar.gz' --exclude=''
        else
            printf 'The model archive is a Git LFS pointer. Install Git LFS and run git lfs pull, then rerun setup.\n' >&2
            exit 1
        fi
    fi
    "$venv_python" - "$archive" "$checksum" <<'PY'
import hashlib
import pathlib
import sys

archive = pathlib.Path(sys.argv[1])
expected = pathlib.Path(sys.argv[2]).read_text().split()[0]
digest = hashlib.sha256()
with archive.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != expected:
    raise SystemExit("Artifact archive SHA-256 does not match its checksum file")
print(f"Archive checksum verified: {digest.hexdigest()}")
PY
    if [[ ! -f "$artifact_root/full_bge_cpu/seed_42/alignment.joblib" ]]; then
        tar -xzf "$archive" -C "$artifact_root"
    fi
elif [[ ! -f "$artifact_root/full_bge_cpu/seed_42/alignment.joblib" ]]; then
    printf 'No trained-model archive or restored artifacts found under %s.\n' "$artifact_root" >&2
    exit 1
fi

printf '\nCPU environment and trained artifacts are ready. No training was run.\n'
printf 'Activate with: source .venv/bin/activate\n'
printf 'Artifacts: %s/full_bge_cpu\n' "$artifact_root"
printf 'Seed summary: %s/full_bge_cpu/seed_summary.json\n' "$artifact_root"