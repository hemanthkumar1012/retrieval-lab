.PHONY: up down test test-sql smoke lint bench bench-quick cache-bench run

up:
	docker compose up -d db redis

down:
	docker compose down

run:
	uvicorn app.main:app --reload

test:
	pytest -q

test-sql: up
	TEST_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag pytest -q tests/test_sql_integration.py

smoke:
	bash scripts/smoke.sh

lint:
	ruff check .

bench-quick: up
	python -m bench.run_bench --limit 100 --strategies recursive_256 fixed_256

bench: up
	python -m bench.run_bench

cache-bench:
	python -m bench.cache_bench
