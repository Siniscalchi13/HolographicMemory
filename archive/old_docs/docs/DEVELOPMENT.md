Development Guide
=================

Local (without Docker)
- make cpp      # build native extension locally
- pip install -e holographic-fs
- uvicorn services.api.app:app --reload --port 8000

Docker Dev
- make build-images
- make dev
- Edit Python files under services/api/ and holographic-fs/ — auto-reloads

Desktop Client
- cd desktop/holo-desktop && npm install
- HOLO_API=http://localhost:8000 npm start

Testing
- Python tests: make test (repo) or make test-ci (container)
- Add API tests under services/api/tests/

C++ Changes
- In dev container: make shell; then re-run build script if necessary

Structure
- services/api: FastAPI app
- holographic-fs: Python library + native bindings loader
- desktop/holo-desktop: Electron client

