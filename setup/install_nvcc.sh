#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
echo "STEP1 apt update"
apt-get update -qq
echo "STEP2 install nvidia-cuda-toolkit"
apt-get install -y -qq nvidia-cuda-toolkit
echo "STEP3 verify"
nvcc --version 2>&1 | tail -3
echo "NVCC_INSTALL_DONE"
