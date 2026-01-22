#!/usr/bin/env bash 

set -euo pipefail

VENV_DIR=".venv"
REQ_FILE="requirements.txt"
CONFIG_FILE="config.yaml"
PY_SCRIPT="training.py"
META_SCRIPT="env_metadata.py"

LOG_DIR="logs"
VIZ_DIR="viz"

echo "==> Initialising Environment"

if [ ! -d "$VENV_DIR" ]; then 
    echo "==> Creating Virtual Environment : $VENV_DIR"
    python -m venv "$VENV_DIR"
else
    echo "==> Virtual Env already exists: $VENV_DIR"
fi 

if [ -f "$VENV_DIR/Scripts/activate" ]; then 
    source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then 
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Could not find venv activate script in $VENV_DIR"
    exit 1
fi 

echo "==> Checking for requirements.txt file "

if [ -f "$REQ_FILE" ]; then 
    echo "==> uprading pip"
    python -m pip install --upgrade pip 
    echo "==> installing dependencies from "
    pip install -r "$REQ_FILE"
else 
    echo "Warning: $REQ_FILE not found."
fi 

if [ -f "$META_SCRIPT" ]; then 
    echo "==> Creating a Metadata file"
    python "$META_SCRIPT"
else
    echo "Please add the Metadata Generation Script $META_SCRIPT"
    exit 1
fi 

