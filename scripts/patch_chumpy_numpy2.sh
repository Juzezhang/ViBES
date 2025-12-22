#!/usr/bin/env bash
set -euo pipefail

echo "[patch-chumpy] Using python: $(command -v python)"
python -c "import sys; print('[patch-chumpy] Python', sys.version.split()[0])"

CHUMPY_INIT="$(python - <<'PY'
import site, pathlib
p = pathlib.Path(site.getsitepackages()[0]) / "chumpy" / "__init__.py"
print(p)
PY
)"

if [[ ! -f "$CHUMPY_INIT" ]]; then
  echo "[patch-chumpy] ERROR: chumpy __init__.py not found at: $CHUMPY_INIT"
  echo "[patch-chumpy] Did you install chumpy in this env?"
  exit 1
fi

echo "[patch-chumpy] Target: $CHUMPY_INIT"
cp -f "$CHUMPY_INIT" "${CHUMPY_INIT}.bak"
echo "[patch-chumpy] Backup: ${CHUMPY_INIT}.bak"

python - <<'PY'
import site, pathlib

p = pathlib.Path(site.getsitepackages()[0]) / "chumpy" / "__init__.py"
txt = p.read_text()

old = "from numpy import bool, int, float, complex, object, unicode, str, nan, inf"
if old not in txt:
    print("[patch-chumpy] Pattern not found; maybe already patched?")
    # still exit 0, because it's likely already patched
    raise SystemExit(0)

new = """try:
    from numpy import bool, int, float, complex, object, unicode, str, nan, inf
except Exception:
    import numpy as _np
    import builtins as _bt
    bool = _np.bool_
    int = _bt.int
    float = _bt.float
    complex = _bt.complex
    object = _bt.object
    unicode = _bt.str
    str = _bt.str
    nan = _np.nan
    inf = _np.inf
"""

p.write_text(txt.replace(old, new))
print("[patch-chumpy] Patched OK:", p)
PY

echo "[patch-chumpy] Verifying import..."
python - <<'PY'
import numpy as np
print("[patch-chumpy] numpy", np.__version__)
import chumpy
print("[patch-chumpy] chumpy import OK")
PY

echo "[patch-chumpy] Done."
