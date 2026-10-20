#!/bin/bash
# ALE-Bench Lite data and tools for verifiable/program_qd/alebench.py, pinned as in the paper's runs: the 7 problems
# from the Hugging Face dataset SakanaAI/ALE-Bench, each problem's Rust tools (gen, tester) built with cargo, and the
# AtCoder Library v1.5.1. Needs curl, unzip, tar, g++ and a Rust toolchain (ALE-Bench builds its tools with Rust 1.79).
#   bash verifiable/setup_alebench.sh      (writes verifiable/third_party/ale-bench and verifiable/third_party/ac-library)
set -euo pipefail
cd "$(dirname "$0")/third_party"
REV=0f426173b4e4e73b09b2b3631ae0490f66b75f99      # alebench.DATA_PIN
mkdir -p ale-bench
for p in ahc008 ahc011 ahc015 ahc024 ahc026 ahc039 ahc046; do
  if [ ! -d "ale-bench/$p" ]; then
    curl -fsSL -o "ale-bench/$p.zip" "https://huggingface.co/datasets/SakanaAI/ALE-Bench/resolve/$REV/$p.zip"
    unzip -q "ale-bench/$p.zip" -d ale-bench && rm "ale-bench/$p.zip"
  fi
  (cd "ale-bench/$p/tools" && cargo build --release --quiet)
done
if [ ! -d ac-library ]; then
  curl -fsSL https://github.com/atcoder/ac-library/archive/refs/tags/v1.5.1.tar.gz | tar xz
  mv ac-library-1.5.1 ac-library
fi
echo "ALE-Bench ready in $(pwd)"
