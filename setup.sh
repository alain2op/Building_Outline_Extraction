
set -e

PY_MIN_MAJOR=3
PY_MIN_MINOR=8
VENV_DIR="venv"

# ---- 1. Python check ----------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "ERROR: Python not found. Install Python ${PY_MIN_MAJOR}.${PY_MIN_MINOR}+ first." >&2
    exit 1
fi

PY_VER=$("$PY" -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')
PY_MAJOR=${PY_VER%%.*}
PY_MINOR=${PY_VER##*.}

if [ "$PY_MAJOR" -lt "$PY_MIN_MAJOR" ] || \
   { [ "$PY_MAJOR" -eq "$PY_MIN_MAJOR" ] && [ "$PY_MINOR" -lt "$PY_MIN_MINOR" ]; }; then
    echo "ERROR: Python ${PY_MIN_MAJOR}.${PY_MIN_MINOR}+ required (found ${PY_VER})." >&2
    exit 1
fi
echo "[1/4] Found Python ${PY_VER}"

# ---- 2. Virtual environment ---------------------------------------
if [ -d "$VENV_DIR" ]; then
    echo "[2/4] Reusing existing venv at ./${VENV_DIR}"
else
    echo "[2/4] Creating venv at ./${VENV_DIR} ..."
    "$PY" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
. "$VENV_DIR/bin/activate"

# ---- 3. Install dependencies --------------------------------------
echo "[3/4] Upgrading pip ..."
python -m pip install --upgrade pip >/dev/null

echo "[3/4] Installing requirements (this can take a minute) ..."
pip install -r requirements.txt

# ---- 4. Done ------------------------------------------------------
echo "[4/4] Setup complete."
echo ""
echo "Next steps:"
echo "  source ${VENV_DIR}/bin/activate     # activate the env in new shells"
echo "  python app.py                       # start the web UI on http://localhost:5000"
echo ""
echo "  # or run from the command line directly:"
echo "  python building_extraction.py path/to/image.png"
echo "  python sentinel_pipeline.py sat_image"
