#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
exec latexmk "Lecture10/ECE408_L10_Introduction_to_ML_Inference_and_Training_in_DNNs_CN.tex" "$@"
