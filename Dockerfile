FROM python:3.11-slim

WORKDIR /srv

# redis-server is only started when REDIS_URL is unset (see scripts/start.sh)
RUN apt-get update \
    && apt-get install -y --no-install-recommends redis-server \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch. The default Linux wheel drags in ~2 GB of CUDA libraries this service never uses.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake both models and the tokenizer vocab into the image so a cold start does not download them.
ENV HF_HOME=/srv/.cache/huggingface \
    TIKTOKEN_CACHE_DIR=/srv/.cache/tiktoken
RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; import tiktoken; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); tiktoken.get_encoding('cl100k_base')"

# some hosts (Hugging Face Spaces) run the container as a non-root user that still needs to
# write lock files into the model cache
RUN chmod -R a+rwX /srv/.cache

COPY app ./app
COPY bench ./bench
COPY scripts/start.sh ./start.sh

# Hosts such as Render, Railway, Fly and Spaces inject PORT or use 8000. The proxy flags in
# start.sh make the client IP (used by the rate limiter) come from X-Forwarded-For, so only
# run this behind a proxy.
ENV PORT=8000
CMD ["sh", "./start.sh"]
