#!/usr/bin/env bash
set -euo pipefail

# Prepare embedded Python.framework + site-packages for the macOS app bundle.
#
# - Downloads Python.org universal2 Python 3.12 framework
# - Installs required wheels into a local site-packages
# - Builds Metal GPU pybind11 module (holographic_gpu) via CMake
# - Places everything under macos/HolographicMemoryApp/Runtime for Xcode to embed

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
APP_DIR="${ROOT_DIR}/macos/HolographicMemoryApp"
RUNTIME_DIR="${APP_DIR}/Runtime"
PY_VER="3.12.5"
PY_PKG="python-${PY_VER}-macos11.pkg"
PY_URL="https://www.python.org/ftp/python/${PY_VER}/${PY_PKG}"

mkdir -p "${RUNTIME_DIR}"

echo "[+] Downloading Python.framework ${PY_VER} (universal2)"
TMPD=$(mktemp -d)
curl -fsSL "${PY_URL}" -o "${TMPD}/${PY_PKG}"
echo "[+] Expanding pkg"
pkgutil --expand-full "${TMPD}/${PY_PKG}" "${TMPD}/py"

echo "[+] Copying Python.framework"
mkdir -p "${RUNTIME_DIR}"
# Newer python.org pkg layout places the Framework at Python_Framework.pkg/Payload/*
if [ -d "${TMPD}/py/Python_Framework.pkg/Payload/Versions" ]; then
  rsync -a "${TMPD}/py/Python_Framework.pkg/Payload/" "${RUNTIME_DIR}/Python.framework/"
elif [ -d "${TMPD}/py/Python_Framework.pkg/Python.framework" ]; then
  rsync -a "${TMPD}/py/Python_Framework.pkg/Python.framework" "${RUNTIME_DIR}/"
else
  echo "[ERR] Could not find Python.framework in expanded pkg" >&2
  exit 1
fi

PY_HOME="${RUNTIME_DIR}/Python.framework/Versions/3.12"
PY_BIN_APP="${RUNTIME_DIR}/Python.framework/Resources/Python.app/Contents/MacOS/Python"
PY_BIN="${PY_HOME}/bin/python3"
# Prefer the Python.app launcher to ensure proper framework resolution
if [[ -x "${PY_BIN_APP}" ]]; then PYTHON="${PY_BIN_APP}"; else PYTHON="${PY_BIN}"; fi

echo "[+] Installing Python packages into embedded runtime"
export DYLD_FRAMEWORK_PATH="${RUNTIME_DIR}"
export PYTHONHOME="${PY_HOME}"
set +e
"${PYTHON}" -m ensurepip --upgrade
"${PYTHON}" -m pip install --upgrade pip wheel setuptools
REQ_FILE="${ROOT_DIR}/services/api/requirements.txt"
"${PYTHON}" -m pip install -r "${REQ_FILE}" && EMBED_OK=1 || EMBED_OK=0
"${PYTHON}" -m pip install numpy Pillow pybind11 || EMBED_OK=0
set -e

DEST_SITE="${APP_DIR}/Runtime/Python/site-packages"
mkdir -p "${DEST_SITE}"

if [[ "${EMBED_OK}" == "1" ]]; then
  echo "[+] Using embedded Python site-packages"
  SITE_PACK=$("${PYTHON}" -c 'import sysconfig;print(sysconfig.get_paths()["platlib"])')
  rsync -a "${SITE_PACK}/" "${DEST_SITE}/"
else
  echo "[!] Embedded pip failed (likely missing SSL). Falling back to host venv."
  python3 -m venv "${TMPD}/hostvenv"
  source "${TMPD}/hostvenv/bin/activate"
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install -r "${REQ_FILE}"
  python -m pip install numpy Pillow pybind11
  HOST_SITE=$(python -c 'import sysconfig;print(sysconfig.get_paths()["platlib"])')
  rsync -a "${HOST_SITE}/" "${DEST_SITE}/"
  deactivate
  # Ensure CPython 3.12-compatible wheels are present (override mismatched binaries)
  echo "[+] Downloading CPython 3.12 arm64 wheels and unpacking into site-packages"
  DL_DIR="${TMPD}/wheels312"
  mkdir -p "${DL_DIR}"
  python3 -m pip download -r "${REQ_FILE}" \
    --platform macosx_11_0_arm64 --only-binary=:all: \
    --python-version 3.12 --implementation cp --abi cp312 \
    -d "${DL_DIR}"
  python3 -m pip download uvicorn[standard] \
    --platform macosx_11_0_arm64 --only-binary=:all: \
    --python-version 3.12 --implementation cp --abi cp312 \
    -d "${DL_DIR}"
  python3 - <<'PY'
import zipfile, os, glob
dl = os.environ['DL_DIR']
dest = os.environ['DEST_SITE']
os.makedirs(dest, exist_ok=True)
ws = glob.glob(os.path.join(dl, '*.whl'))
for w in ws:
    with zipfile.ZipFile(w) as z:
        z.extractall(dest)
print('[+] Unpacked %d wheels into %s' % (len(ws), dest))
PY
fi

echo "[+] Building Metal GPU module (holographic_gpu) for CPython 3.12"
BUILD_DIR="${TMPD}/build_gpu"
mkdir -p "${BUILD_DIR}"

# Locate pybind11 CMake glue from host Python (fallback to embedded)
PYCMAKE_DIR=$(python3 -m pybind11 --cmakedir 2>/dev/null || true)
if [[ -z "${PYCMAKE_DIR}" ]]; then
  PYCMAKE_DIR=$("${PYTHON}" -m pybind11 --cmakedir 2>/dev/null || true)
fi

export DYLD_FRAMEWORK_PATH="${RUNTIME_DIR}"
cmake -S "${ROOT_DIR}/holographic-fs/native/holographic" -B "${BUILD_DIR}" \
  -DBUILD_METAL=ON -DBUILD_CUDA=OFF -DBUILD_ROCM=OFF \
  -Dpybind11_DIR="${PYCMAKE_DIR}" \
  -DPYTHON_EXECUTABLE="${PY_BIN_APP}" \
  -DPYTHON_INCLUDE_DIR="${PY_HOME}/include/python3.12" \
  -DPYTHON_LIBRARY="${PY_HOME}/Python"

cmake --build "${BUILD_DIR}" --config Release -j

echo "[+] Copying GPU module"
cp -f "${BUILD_DIR}/"holographic_gpu*.so "${DEST_SITE}/" || {
  echo "[WARN] holographic_gpu build failed or not found; continuing without GPU module"; true; }

echo "[+] Prepared runtime under: ${RUNTIME_DIR}"
echo "    - Python.framework
    - Python/site-packages
    - holographic_gpu*.so (if built)"

rm -rf "${TMPD}"
