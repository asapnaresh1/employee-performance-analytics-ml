.PHONY: install generate train lint format type-check test api dashboard docker-up docker-down clean all

install:
	pip install -r requirements.txt -r requirements-dev.txt

generate:
	python -m src.data.generator

train:
	python -m src.models.predictor

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

type-check:
	mypy src/

test:
	pytest tests/ -v --tb=short

api:
	uvicorn src.api.main:app --reload

dashboard:
	streamlit run src/dashboard/app.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf data/raw/*.csv data/processed/*.csv models/*.joblib

all: generate train test
