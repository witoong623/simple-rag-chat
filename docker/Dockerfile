FROM python:3.11-slim-bookworm

ENV TZ=Asia/Bangkok DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["streamlit", "run", "main.py", "/app/config/config.yaml"]
