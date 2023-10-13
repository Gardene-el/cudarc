#!/bin/bash
set -exu

bindgen \
  --allowlist-type="^cuda.*" \
  --allowlist-var="^cuda.*" \
  --allowlist-function="^cuda.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  wrapper.h -- -I/opt/cuda/include \
  > sys.rs