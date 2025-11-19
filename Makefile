.PHONY: install test lint format type-check clean run-hardware run-cre dashboard

install:
	pip install -r requirements.txt

test:
	export PYTHONPATH=src:. && pytest tests/

lint:
	ruff check src tests

format:
	ruff format src tests

type-check:
	mypy src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-hardware:
	export PYTHONPATH=src:. && python -m icp.cli.score_accounts --division hardware

run-cre:
	export PYTHONPATH=src:. && python -m icp.cli.score_accounts --division cre

dashboard:
	export PYTHONPATH=src:. && streamlit run apps/dashboard.py

weights:
	export PYTHONPATH=src:. && python -m icp.cli.generate_weights

score:
	export PYTHONPATH=src:. && python -m icp.cli.score_accounts

playbooks:
	export PYTHONPATH=src:. && python -m icp.cli.build_playbooks

call-lists:
	export PYTHONPATH=src:. && python -m icp.cli.export_call_lists

pipeline: score playbooks call-lists
