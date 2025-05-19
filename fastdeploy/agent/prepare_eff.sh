#!/bin/bash

source ./fastdeploy/agent/build_env_eff.sh

# 1. Check if virtual environment exists
if [ ! -d "$FASTDEPLOY_ENV_NAME" ]; then
    echo "[1/4] Virtual environment '$FASTDEPLOY_ENV_NAME' does not exist, creating..."

    # 2. Create virtual environment with specified Python version
    if ! command -v "python$FASTDEPLOY_PYTHON_VERSION" &> /dev/null; then
        echo "Error: Python$FASTDEPLOY_PYTHON_VERSION not found!"
        echo "Please take one of the following actions:"
        echo "1. Install Python$FASTDEPLOY_PYTHON_VERSION"
        exit 1
    fi

    echo "[2/4] Creating virtual environment with Python $FASTDEPLOY_PYTHON_VERSION..."
    if ! "python$FASTDEPLOY_PYTHON_VERSION" -m virtualenv "$FASTDEPLOY_ENV_NAME"; then
        echo "Failed to create virtual environment!"
        rm -fr $FASTDEPLOY_ENV_NAME
        exit 1
    fi
else
    echo "[1/4] Virtual environment '$FASTDEPLOY_ENV_NAME' already exists"
fi

# 3. Activate virtual environment
echo "[3/4] Activating virtual environment..."
source "$FASTDEPLOY_ENV_NAME/bin/activate"

# Check exit status of previous command
if [ $? -eq 0 ]; then
    echo "Virtual environment activated successfully."
else
    echo "Failed to activate virtual environment."
    exit 1
fi

# 4. Check and install dependencies
echo "[4/4] Checking dependencies..."
if [ -f "$FASTDEPLOY_REQUIREMENTS" ]; then
    echo "Dependencies file detected, installing..."
    pip install --upgrade pip
    pip install -r "$FASTDEPLOY_REQUIREMENTS"
else
    echo "Dependencies file $FASTDEPLOY_REQUIREMENTS not found, skipping installation"
fi

echo "[Done] Environment setup completed!"
