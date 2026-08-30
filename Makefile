# Common tasks. Run `make` on its own to list them.
#
# These assume an activated virtualenv (see `make install`).

PYTHON ?= python
STREAMLIT_PORT ?= 8501

.DEFAULT_GOAL := help
.PHONY: help install install-fast install-dev run test test-fast bench bench-solvers screenshots diagram demo clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-fast: ## Install the optional Boykov-Kolmogorov solver
	$(PYTHON) -m pip install -r requirements-fast.txt

install-dev: ## Install screenshot/demo tooling (Playwright + imageio)
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m playwright install chromium

run: ## Launch the Streamlit app
	$(PYTHON) -m streamlit run app.py --server.port $(STREAMLIT_PORT)

test: ## Run the test suite
	$(PYTHON) -m pytest

test-fast: ## Run the suite including the fast-mode tests (needs install-fast)
	$(PYTHON) -m pytest -v

bench: ## Measure end-to-end segmentation time
	$(PYTHON) benchmarks/measure_timing.py

bench-solvers: ## Compare Edmonds-Karp against Boykov-Kolmogorov (needs install-fast)
	$(PYTHON) benchmarks/compare_solvers.py

screenshots: ## Recapture the app screenshots and demo GIF (needs install-dev)
	$(PYTHON) scripts/capture_screenshots.py

diagram: ## Regenerate the README graph diagram
	$(PYTHON) scripts/make_diagram.py

demo: ## Segment a sample photo from the command line
	$(PYTHON) -m segmentation.cli sample_images/horse.jpg out.png \
		--fg 350,300,250,20 --bg 60,40,340,20 --bg 100,740,500,20

clean: ## Remove caches and generated scratch files
	rm -rf .pytest_cache screenshots/_frames out.png
	find . -name '__pycache__' -type d -not -path './venv/*' -prune -exec rm -rf {} +
