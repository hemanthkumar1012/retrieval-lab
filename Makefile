.PHONY: up down test lint bench bench-quick cache-bench run

up:
	docker compose up -d db redis

down:
	docker compose down

run:
	uvicorn app.main:app --reload

test:
	pytest -q

lint:
	ruff check .

bench-quick: up
	python -m bench.run_bench --limit 100 --strategies recursive_256 fixed_256

bench: up
	python -m bench.run_bench

cache-bench:
	python -m bench.cache_bench
