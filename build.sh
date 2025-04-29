#!/bin/bash

# conda create -yn xattn python=3.10
# conda activate xattn

# conda install -y git
# conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit
# conda install -y nvidia::cuda-cudart-dev
# conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
uv pip install --upgrade pip
uv pip install torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0 psutil
uv pip install --no-build-isolation \
  transformers==4.46 \
  accelerate \
  sentencepiece \
  minference==0.1.5.post1 \
  datasets \
  wandb \
  zstandard \
  matplotlib \
  huggingface_hub==0.23.2 \
  xformers \
  vllm==0.6.3.post1 \
  vllm-flash-attn==2.6.1 \
  tensor_parallel==2.0.0 \
  ninja \
  packaging \
  ray==2.40.0

uv pip install flash-attn==2.6.3 --no-build-isolation --verbose
# uv pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ --verbose
# uv pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/ --verbose
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/ --verbose

# LongBench evaluation
uv pip install seaborn rouge_score einops pandas


# Install xAttention
uv pip install -e . --verbose --no-build-isolation

# Install Block Sparse Streaming Attention
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
uv pip install -e . --verbose --no-build-isolation
cd ..

export PYTHONPATH="$PYTHONPATH:$(pwd)"
