#!/usr/bin/env bash
# One-time codespace setup (same steps as class). Run with: bash setup.sh
set -e

pip install --user llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install --user -U huggingface_hub

hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir ./models
