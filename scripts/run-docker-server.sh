#!/bin/env bash

docker run -d --rm --gpus all --name llama-server \
    -p 8000:8000 \
    -v ${PWD}/models:/models \
    local/llama.cpp:server-cuda \
    -m /models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf \
    --port 8000 --host 0.0.0.0 -n 512 --n-gpu-layers 3
