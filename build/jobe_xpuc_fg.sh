#!/bin/bash
cd /vxk/build
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
echo "=== cmake --build --target _xpu_C -j2 ==="
cmake --build . --target _xpu_C -j2 2>&1 | tail -70
echo "=== XPUC_BUILD_EXIT=${PIPESTATUS[0]} ==="
echo "FOREGROUND_DONE"
