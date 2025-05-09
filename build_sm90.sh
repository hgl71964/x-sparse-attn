#!/bin/bash
# minimal build for hopper
rm -rf .venv
uv venv .venv --seed --python 3.10
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install packaging ninja pytest
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# AS OF MAY 8TH, 2025, WE NEED TO USE TORCH 2.4 W/ CUDA 12.4 FOR FLASHINFER COMPATIBILITY
uv pip install torch==2.4.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 

uv pip install transformers==4.46 accelerate
# flashinfer build not working well with uv for some reason?
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/ --verbose

# Install xAttention
uv pip install -e . --verbose --no-build-isolation

# # Install Block Sparse Streaming Attention
git clone git@github.com:mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
export FLASH_ATTENTION_FORCE_BUILD="TRUE"
export TORCH_CUDA_ARCH_LIST="9.0"
uv pip install -e . --verbose --no-build-isolation
cd ..
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# flash attn
git submodule init
git submodule update
cd ./third-party/flash-attention/hopper
python setup.py install
export PYTHONPATH="$PYTHONPATH:$(pwd)"
cd ../../..

# Below will benchmark speedup (XAttention only, FlexPrefill and MInference will not work with this build)
python eval/efficiency/attention_speedup_flashattn.py 