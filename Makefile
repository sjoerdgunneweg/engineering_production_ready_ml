.PHONY: test
test:
	coverage run -m pytest -v . && coverage report --rcfile=.coveragerc

.PHONY: format
format:
	ruff format .
