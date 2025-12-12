.PHONY: test
test:
	docker compose -f infra/test-compose.yaml run --rm --remove-orphans pytest

.PHONY: format
format:
	ruff format . --line-length 120

.PHONY: build
build:
	docker build . -t alcoholerometer -f infra/build/Dockerfile

.PHONY: up
up: build
	docker compose --file infra/docker-compose.yaml down -v
	docker compose -f infra/docker-compose.yaml up --remove-orphans

.PHONY: logs
logs:
	docker compose -f infra/docker-compose.yaml logs -f