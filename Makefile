PY?=python
PIP?=$(PY) -m pip

.PHONY: setup test build api openapi clean cpp

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
	uvicorn services.api.app:app --reload --port 8080

openapi:
	$(PY) scripts/dump_openapi.py

clean:
	rm -rf **/__pycache__ build dist .pytest_cache holographic-fs/*.egg-info holographic-fs/native/holographic/build

