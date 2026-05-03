.PHONY: install lint fmt

install:
	pip install -r requirements.txt

lint:
	ruff check src/
	black --check src/

fmt:
	black src/
	ruff check --fix src/
