PY?=python
PIP?=$(PY) -m pip

.PHONY: setup test build api openapi clean cpp launch stop \
	build-images dev prod test-ci logs shell down deploy

setup:
	cd holographic-fs && $(PIP) install -e . && $(PIP) install -r dev-requirements.txt

test:
	cd holographic-fs && $(PY) -m pytest -q

build:
	cd holographic-fs && $(PY) -m build

cpp:
	$(PIP) install pybind11
	$(PY) -c "import sys,os; p='holographic-fs/native/holographic'; os.chdir(p); print('Building C++ in', os.getcwd()); import subprocess; subprocess.check_call([sys.executable,'setup.py','build_ext','--inplace'])"

api:
	$(PY) -m pip install -r services/api/requirements.txt
	uvicorn services.api.app:app --reload --port 8000

openapi:
	$(PY) scripts/dump_openapi.py

launch:
	./launch_holo.sh

stop:
	@echo "🛑 Stopping HolographicMemory processes..."
	@pkill -f "uvicorn.*app:app" 2>/dev/null || echo "No API processes running"
	@echo "✅ All processes stopped"

clean:
	rm -rf **/__pycache__ build dist .pytest_cache holographic-fs/*.egg-info holographic-fs/native/holographic/build

# -------- Enterprise Docker targets --------

build-images:
	docker compose build

dev:
	docker compose up -d --remove-orphans
	@echo "API: http://localhost:8000 (key: $$HOLO_API_KEY)"

prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build --remove-orphans
	@echo "API: http://localhost:8000 (prod)"

test-ci:
	docker compose run --rm api pytest -q -k "not slow" || true

logs:
	docker compose logs -f api

shell:
	docker compose exec api bash || docker compose exec api sh

down:
	docker compose down -v --remove-orphans

deploy:
	@echo "For Kubernetes, see docs/DEPLOYMENT.md and deploy/k8s/*.yaml"
