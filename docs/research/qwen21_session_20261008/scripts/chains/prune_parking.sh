#!/bin/bash
# Trainer parking dirs are created per park/wake and never removed; 46k of them (820 GB) filled the NVMe
# on 2026-10-09. Dirs untouched for an hour belong to finished phases.
while true; do
  find /mnt/nvme/outputs/qwen21_aesthetic_shared_noise/parking -mindepth 1 -maxdepth 1 -type d -mmin +60 -print0 2>/dev/null | xargs -0 -r rm -rf
  sleep 600
done
