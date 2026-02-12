#!/usr/bin/env bash 

set -euo pipefail

VENV_DIR=".venv"
REQ_FILE="requirements.txt"
CONFIG_FILE="config.yaml"
PY_SCRIPT="main.py"

LOG_DIR="logs"
VIZ_DIR="viz"

echo "###################################################################"
echo "==> Initialising Environment"

if [ ! -d "$VENV_DIR" ]; then 
    echo "==> Creating Virtual Environment : $VENV_DIR"
    python -m venv "$VENV_DIR"
else
    echo "==> Virtual Env already exists: $VENV_DIR"
fi 
echo "###################################################################"

if [ -f "$VENV_DIR/Scripts/activate" ]; then 
    source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then 
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Could not find venv activate script in $VENV_DIR"
    exit 1
fi 

echo "###################################################################"

echo "==> Checking for requirements.txt file "

if [ -f "$REQ_FILE" ]; then 
    echo "###################################################################"
    echo "==> uprading pip"
    python -m pip install --upgrade pip 
    echo "==> installing dependencies from "
    pip install -r "$REQ_FILE"
    echo "###################################################################"
else 
    echo "Warning: $REQ_FILE not found."
fi 





echo "--> Creating Output folders of Logs and Visualizations"
mkdir -p "$LOG_DIR"
mkdir -p "$VIZ_DIR"
echo "###################################################################"


if [ -f "$PY_SCRIPT" ]; then 
    echo "###################################################################"
    echo "==> Running the Main Training Script."
    python "$PY_SCRIPT"
    echo "###################################################################"
else
    echo "Please add the repro_check Script $PY_SCRIPT"
    exit 1
    echo "###################################################################"

fi 