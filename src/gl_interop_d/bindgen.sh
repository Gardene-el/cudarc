#!/bin/bash
set -exu

bindgen \
  --allowlist-type="^CUGL.*" \
  --allowlist-type="^CU.*" \
  --allowlist-function="^cuGL.*" \
  --allowlist-var="^CU_GL.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  wrapper.h -- -I/opt/cuda/include \
  > sys.rs