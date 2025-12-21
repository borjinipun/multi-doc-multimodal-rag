install:
	python -m venv .venv && . .venv/bin/activate && pip install -e .

lint:
	flake8 src

test:
	pytest -q

run-example:
	mdrag ingest --data-dir data --persist-dir doc_db
