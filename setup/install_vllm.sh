#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
echo "STEP1 apt update"
apt-get update -qq
echo "STEP2 apt install python3-venv python3-pip"
apt-get install -y -qq python3-venv python3-pip
echo "STEP3 create venv"
su - luxing -c 'python3 -m venv ~/venvs/vllm'
echo "STEP4 upgrade pip"
su - luxing -c '~/venvs/vllm/bin/pip install -q --upgrade pip'
echo "STEP5 pip install vllm"
su - luxing -c '~/venvs/vllm/bin/pip install vllm'
echo "STEP6 verify"
su - luxing -c '~/venvs/vllm/bin/python -c "import vllm; print(vllm.__version__)"'
echo "VLLM_INSTALL_OK"
