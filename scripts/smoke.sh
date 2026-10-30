#!/usr/bin/env bash
# End-to-end smoke test against a running API (default http://localhost:8000):
# ingests three documents, runs every retrieval mode, checks /ask streams.
# Uses the fixed_128 strategy so the default one is left alone; it does leave
# three smoke-* documents behind.
set -euo pipefail
BASE=${1:-http://localhost:8000}
STRATEGY=fixed_128
QUERY="does vitamin D reduce fracture risk"

curl -sf "$BASE/health" >/dev/null
# the page and its scripts are served by the API itself
curl -sf "$BASE/" | grep -q "retrieval-lab"
for f in app.js styles.css; do curl -sf "$BASE/static/$f" >/dev/null; done

curl -sf -X POST "$BASE/ingest" -H 'content-type: application/json' -d '{
  "strategy": "'"$STRATEGY"'",
  "docs": [
    {"id": "smoke-1", "title": "Vitamin D and fractures",
     "text": "Vitamin D supplementation reduces fracture risk in elderly adults, according to a pooled analysis of randomized trials."},
    {"id": "smoke-2", "title": "Calcium and bone",
     "text": "Calcium intake and bone density were measured in postmenopausal women over five years."},
    {"id": "smoke-3", "title": "Statins",
     "text": "Statin therapy lowers cardiovascular mortality in patients with elevated cholesterol."}
  ]}' | python3 -c 'import json, sys; d = json.load(sys.stdin); assert d["chunks"] >= 3, d; print("ingested", d)'

for mode in dense bm25 hybrid hybrid_rerank; do
  curl -sf -X POST "$BASE/search" -H 'content-type: application/json' \
    -d "{\"query\": \"$QUERY\", \"mode\": \"$mode\", \"strategy\": \"$STRATEGY\", \"k\": 3}" |
    python3 -c "
import json, sys
d = json.load(sys.stdin)
top = d['hits'][0]['doc_id']
assert top == 'smoke-1', ('$mode', top, d['hits'])
print('$mode ok', d['timings_ms'])"
done

# without LLM_API_KEY the endpoint answers with a notice, but must still stream meta and done
out=$(curl -sfN -X POST "$BASE/ask" -H 'content-type: application/json' \
  -d "{\"query\": \"$QUERY\", \"strategy\": \"$STRATEGY\", \"use_cache\": false}")
grep -q 'event: meta' <<<"$out"
grep -q 'event: done' <<<"$out"
echo "/ask streamed meta and done"

echo "smoke test passed"
