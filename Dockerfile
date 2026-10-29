FROM python:3.11-slim

WORKDIR /srv

# CPU-only torch. The default Linux wheel drags in ~2 GB of CUDA libraries this service never uses.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake both models and the tokenizer vocab into the image so a cold start does not download them.
ENV HF_HOME=/srv/.cache/huggingface \
    TIKTOKEN_CACHE_DIR=/srv/.cache/tiktoken
RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; import tiktoken; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); tiktoken.get_encoding('cl100k_base')"

COPY app ./app
COPY bench ./bench

# Hosts such as Render, Railway and Fly inject PORT. The proxy flags make the client IP
# (used by the rate limiter) come from X-Forwarded-For, so only run this behind a proxy.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
