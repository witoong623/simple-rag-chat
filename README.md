# Simple RAG chat
Experiment with Langchain to build converational RAG Q&A application.

The application can use any LLM that can be hosted by OpenAI compatible API server.

# Deployment
## Deployment with local LLM server
### Start llama.cpp server
1. Download `.gguf` model from Hugging Face. I use [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) Q6_K_L model, and put it in `models` directory.
2. Edit `scripts/run-docker-server.sh` script and change the image name of llama.cpp to the version you want to use. I build it locally, so I use `local/llama.cpp:server-cuda`. Generally, the server image is enough for running e.g. `ghcr.io/ggerganov/llama.cpp:server`
3. Edit the model name from `Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf` to the model name that you download into `models` directory.
4. Modify the number of layer to offlaod to GPU by editing argument `--n-gpu-layers 3` to different value, or delete it in case you don't have GPU.
5. Run the script using `./scripts/run-docker-server.sh`.

The server is available at `http://localhost:8000`.
