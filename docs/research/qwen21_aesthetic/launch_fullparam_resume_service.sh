#!/usr/bin/env bash
# Resume acceptance belongs in a separate capped service, never the terminal scope.
set -euo pipefail
systemd-run --user --unit=qwen21-aesthetic-resume-check \
  --property=MemoryHigh=280G --property=MemoryMax=300G --property=MemorySwapMax=0 \
  --property=OOMPolicy=kill --property=KillMode=control-group \
  --property=TimeoutStopSec=30 --property=Restart=no \
  --property=StandardOutput=append:/mnt/nvme/outputs/qwen21_aesthetic_fullparam/resume_check/service.log \
  --property=StandardError=inherit \
  /mnt/nvme/outputs/qwen21_aesthetic_fullparam/resume_check/launch.sh
