FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libfftw3-dev && rm -rf /var/lib/apt/lists/*

# Copy library and build C++ extensions
COPY holographic-fs /app/holographic-fs
RUN python -m pip install --upgrade pip && \
    python -m pip install pybind11 && \
    python -m pip install -e /app/holographic-fs && \
    python -c "import sys,os; p='/app/holographic-fs/native/holographic'; os.chdir(p); import subprocess; subprocess.check_call([sys.executable,'setup.py','build_ext','--inplace'])"

# API deps
COPY services/api/requirements.txt /app/services/api/requirements.txt
RUN python -m pip install -r /app/services/api/requirements.txt

# Copy API code
COPY services/api /app/services/api

ENV HOLO_ROOT=/data
ENV GRID_SIZE=64
EXPOSE 8080

CMD ["uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "8080"]

