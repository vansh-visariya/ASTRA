.PHONY: help install test lint demo clean docker-build docker-run experiments

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run unit tests"
	@echo "  make lint          - Run linting"
	@echo "  make demo          - Run demo experiment"
	@echo "  make clean         - Clean generated files"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make experiments   - Run all experiments"

install:
	pip install -r requirements.txt

test:
	pytest federated/tests/ -v --tb=short

lint:
	black --check federated/ main.py || true
	flake8 federated/ main.py || true

demo:
	python main.py --config config.yaml --demo --seed 42

clean:
	rm -rf runs/
	rm -rf __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t async-fl:latest .

docker-run:
	docker run --gpus all -v $(pwd):/app async-fl:latest

experiments:
	python federated/experiments/run_experiment.py --output ./experiments_results

benchmark:
	python -c "
import time
import torch
from federated.model_zoo import create_model

model = create_model({'model': {'type': 'cnn', 'cnn': {'name': 'simple_cnn'}}})
data = torch.randn(32, 1, 28, 28)

start = time.time()
for _ in range(100):
    output = model(data)
end = time.time()
print(f'Time per forward pass: {(end-start)/100*1000:.2f}ms')
"
