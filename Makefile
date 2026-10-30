.PHONY: install lint format test eval terraform-init terraform-plan adk-run agents-cli-deploy

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff

install:
	pip install -e ".[dev]"

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) check --fix .
	$(RUFF) format .

test:
	$(PYTEST) -v

eval:
	$(PYTHON) -m evals.run_evals

terraform-init:
	cd deployment/terraform && terraform init -backend=false

terraform-plan:
	cd deployment/terraform && terraform plan

adk-run:
	.venv/bin/adk run app

agents-cli-deploy:
	agents-cli deploy --target cloud_run --agent-dir app
