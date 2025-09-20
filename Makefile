.PHONY: setup clean test run

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	pre-commit install

data-analysis:
	jupyter notebook notebooks/01_eda_analysis.ipynb

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

test:
	pytest tests/

format:
	black src/ tests/
	isort src/ tests/

lint:
	pylint src/
	flake8 src/