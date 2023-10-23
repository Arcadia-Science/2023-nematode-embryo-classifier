
.PHONY: setup-develop
setup-develop:
	pip install -e .'[dev]'
	pre-commit install

.PHONY: lint
lint:
	flake8 . --count --statistics --exit-zero
	black --check src

.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: test
test:
	pytest -v
