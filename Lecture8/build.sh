#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
command -v latexmk >/dev/null 2>&1 || {
    printf '%s\n' 'latexmk and XeLaTeX are required.' >&2
    exit 1
}
exec latexmk Lecture8/ECE408_L08_Convolution_Concept_Constant_Cache_CN.tex
