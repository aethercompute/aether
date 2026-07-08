#!/usr/bin/env bash
set -euo pipefail

python_bin="${AETHER_PYTHON:-python3.12}"
torch_version="${TORCH_VERSION:-2.9.1}"
torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  cat >&2 <<EOF
error: $python_bin was not found

Install Python 3.12, or set AETHER_PYTHON to the Python interpreter that has
the project's PyTorch install. Example:

  AETHER_PYTHON=/path/to/python just ci-local
EOF
  exit 1
fi

export AETHER_PYTHON_BIN="$python_bin"
export AETHER_TORCH_VERSION="$torch_version"
export AETHER_TORCH_INDEX_URL="$torch_index_url"

native_libs="$(LD_LIBRARY_PATH= "$python_bin" <<'PY'
import os
import sys
import sysconfig

try:
    import torch
except Exception as exc:
    print(f"error: failed to import torch: {exc}", file=sys.stderr)
    print("Install the CI-matching PyTorch build with:", file=sys.stderr)
    print(
        f"  {os.environ['AETHER_PYTHON_BIN']} -m pip install torch=={os.environ['AETHER_TORCH_VERSION']} --index-url {os.environ['AETHER_TORCH_INDEX_URL']}",
        file=sys.stderr,
    )
    sys.exit(1)

torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if not os.path.isdir(torch_lib):
    print(f"error: torch lib directory does not exist: {torch_lib}", file=sys.stderr)
    sys.exit(1)

python_lib = sysconfig.get_config_var("LIBDIR") or ""
print(torch_lib + ((":" + python_lib) if python_lib else ""))
PY
)"

python_libdir="$(LD_LIBRARY_PATH= "$python_bin" <<'PY'
import sysconfig

print(sysconfig.get_config_var("LIBDIR") or "")
PY
)"

python_link_dir="$(LD_LIBRARY_PATH= "$python_bin" <<'PY'
import os
import sysconfig

libdir = sysconfig.get_config_var("LIBDIR") or ""
ldlibrary = sysconfig.get_config_var("LDLIBRARY") or ""
instsoname = sysconfig.get_config_var("INSTSONAME") or ""

if libdir and ldlibrary and not os.path.exists(os.path.join(libdir, ldlibrary)):
    soname_path = os.path.join(libdir, instsoname)
    if instsoname and os.path.exists(soname_path):
        print(ldlibrary)
        print(soname_path)
PY
)"

if [[ -n "$python_link_dir" ]]; then
  mapfile -t python_link_parts <<< "$python_link_dir"
  python_ldlibrary="${python_link_parts[0]}"
  python_soname_path="${python_link_parts[1]}"
  mkdir -p target/python-libs
  ln -sf "$python_soname_path" "target/python-libs/$python_ldlibrary"
  export LIBRARY_PATH="$(pwd)/target/python-libs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi

if [[ -n "$python_libdir" ]]; then
  export LIBRARY_PATH="$python_libdir${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi

export LIBTORCH_USE_PYTORCH="${LIBTORCH_USE_PYTORCH:-1}"
export PYO3_PYTHON="${PYO3_PYTHON:-$python_bin}"
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY="${PYO3_USE_ABI3_FORWARD_COMPATIBILITY:-1}"

# Drop stale torch library paths from other Python installs. Keeping them in
# LD_LIBRARY_PATH can make test binaries load mismatched torch shared objects.
clean_ld_library_path=""
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  IFS=":" read -ra ld_paths <<< "$LD_LIBRARY_PATH"
  for path in "${ld_paths[@]}"; do
    if [[ "$path" == */site-packages/torch/lib ]]; then
      continue
    fi
    clean_ld_library_path="${clean_ld_library_path:+${clean_ld_library_path}:}${path}"
  done
fi

export LD_LIBRARY_PATH="${native_libs}${clean_ld_library_path:+:${clean_ld_library_path}}"

exec "$@"
