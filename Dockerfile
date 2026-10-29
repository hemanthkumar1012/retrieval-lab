FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt requirements-bench.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-bench.txt

# the tokenizer vocab used for chunking, so ingesting does not need the network later
ENV TIKTOKEN_CACHE_DIR=/srv/.cache/tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" && chmod -R a+rwX /srv/.cache

COPY app ./app
COPY bench ./bench
COPY models ./models

# Hosts such as Render, Railway and Fly inject PORT. The proxy flags make the client IP
# (used by the rate limiter) come from X-Forwarded-For, so only run this behind a proxy.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
