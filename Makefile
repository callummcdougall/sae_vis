format:
	poetry run ruff format .
	poetry run ruff check --fix-only .

lint:
	poetry run ruff check .
	poetry run ruff format --check .

test:
	poetry run pytest

check-all:
	make format
	make lint
	make test
