.PHONY: test
test:
	coverage run -m pytest -v -p no:warnings . && coverage report --rcfile=.coveragerc

.PHONY: format
format:
	ruff format . --line-length 120

.PHONY: build
build:
	docker build . -t alcoholerometer -f infra/build/Dockerfile

.PHONY: up
up: build
	docker compose --file infra/docker-compose.yaml down
	docker compose --file infra/docker-compose.yaml up app

.PHONY: logs
logs:
	docker compose -f infra/docker-compose.yaml logs -f app