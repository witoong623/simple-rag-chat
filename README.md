# Simple RAG chat
Experiment with Langchain to build converational RAG Q&A application.

The application can use any LLM that can be hosted by OpenAI compatible API server.

# Deployment
## Deployment with OpenAI API key
1. Edit `config.yaml` by set `openai_api_key` to OpenAI API key, and remove value `openai_compatible_base_url` (make it empty value).
2. From root directory of the project, run `docker compose -f docker/docker-compose.yaml up -d`.
3. Access the web via port 8501 (you can edit the port mapping in `docker/docker-compose.yaml`).

## Deployment with local LLM server
### Start llama.cpp server
1. Download `.gguf` model from Hugging Face. I use [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) Q6_K_L model, and put it in `models` directory.
2. Edit `docker/docker-compose.llama-cpp.yaml` file to config how llama.cpp server run as follow.
    - In `image` attribute, set it to image version you want to use. I build it locally, so I use `local/llama.cpp:server-cuda`. Generally, the server image is enough for running e.g. `ghcr.io/ggerganov/llama.cpp:server`
    - In `volumes` attribute, add bind mount as necessary to add document to use for RAG.
    - In `command` attribute, change `Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf` to the model name that you download into `models` directory.
    - In `command` attribute, change any necessary you want to config the server.
3. In `docker/docker-compose.yaml` file, edit `volumes` attribute to add bind mount as necessary to add document to use for RAG.
4. Edit `config.yaml` file to config how the app run as follow
    - Set `openai_api_key` to any string (but not empty string).
    - Set `openai_compatible_base_url` to `http://llama-cpp-server:8000`.
    - Set `document_path` to the path of the document that you add to bind mount.
5. From root directory of the project, run `docker compose -f docker/docker-compose.yaml -f docker/docker-compose.llama-cpp.yaml up -d`.
