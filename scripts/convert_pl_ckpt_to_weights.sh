#!/usr/bin/env bash
set -euo pipefail

IN_CKPT="${1:-}"
OUT_CKPT="${2:-}"
MODE="${3:-none}"   # none | copy | symlink | replace

if [[ -z "${IN_CKPT}" || -z "${OUT_CKPT}" ]]; then
  echo "Usage: $0 <IN_CKPT> <OUT_CKPT> [none|copy|symlink|replace]"
  exit 2
fi

mkdir -p "$(dirname "${OUT_CKPT}")"

export IN_CKPT OUT_CKPT MODE

python - <<'PY'
import os
import torch

in_ckpt = os.environ["IN_CKPT"]
out_ckpt = os.environ["OUT_CKPT"]

# 尽量用 weights_only=True（能避开 missing module 的 unpickle 问题）
try:
    ckpt = torch.load(in_ckpt, map_location="cpu", weights_only=True)
except TypeError:
    ckpt = torch.load(in_ckpt, map_location="cpu")

state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
torch.save(state, out_ckpt)

n = len(state) if hasattr(state, "__len__") else -1
print(f"[OK] extracted weights -> {out_ckpt} (keys={n})")
PY

# 可选：把输入 ckpt 替换成权重版（保持你原代码路径不变）
if [[ "${MODE}" != "none" ]]; then
  BAK="${IN_CKPT}.bak"
  if [[ -e "${BAK}" ]]; then
    # 避免重复覆盖旧备份
    BAK="${IN_CKPT}.bak.$(date +%Y%m%d_%H%M%S)"
  fi

  if [[ "${MODE}" == "replace" ]]; then
    mv "${IN_CKPT}" "${BAK}"
    cp -f "${OUT_CKPT}" "${IN_CKPT}"
    echo "[OK] replace: ${IN_CKPT} (backup -> ${BAK})"
  elif [[ "${MODE}" == "copy" ]]; then
    cp -f "${OUT_CKPT}" "${IN_CKPT}"
    echo "[OK] copy weights to: ${IN_CKPT}"
  elif [[ "${MODE}" == "symlink" ]]; then
    mv "${IN_CKPT}" "${BAK}"
    ln -sfn "${OUT_CKPT}" "${IN_CKPT}"
    echo "[OK] symlink: ${IN_CKPT} -> ${OUT_CKPT} (backup -> ${BAK})"
  else
    echo "[WARN] unknown MODE=${MODE}, supported: none|copy|symlink|replace"
    exit 3
  fi
fi
