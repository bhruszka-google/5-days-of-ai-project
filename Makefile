.PHONY: install lint format test eval-validate eval eval-adk terraform-init terraform-plan adk-run agents-cli-deploy

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
ADK := .venv/bin/adk

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

eval-validate:
	$(PYTHON) -m evals.run_evals --validate-only

eval:
	$(PYTHON) -m evals.run_evals --confirm-live-eval --mode live --evalset-path evals/autodeal_golden.evalset.json --config-path evals/eval_config.json

eval-adk:
	$(ADK) eval app evals/autodeal_golden.evalset.json --config_file_path evals/eval_config.json --print_detailed_results

terraform-init:
	cd deployment/terraform && terraform init -backend=false

terraform-plan:
	cd deployment/terraform && terraform plan

adk-run:
	.venv/bin/adk run app

agents-cli-deploy:
	agents-cli deploy --target cloud_run --agent-dir app
