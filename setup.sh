#!/usr/bin/env bash

# This script sets up a virtual environment using uv and pip,
# then installs dependencies from requirements.txt and the package itself in development mode.

# Name of the virtual environment directory
VENV_DIR=".venv"

# Create virtual environment using uv
if ! command -v uv &> /dev/null
then
    echo "uv is not installed. Installing uv with pip..."
    pip install uv || {
      echo "Failed to install uv. Exiting."
      exit 1
    }
fi

echo "Creating virtual environment..."
uv venv --python=3.12 $VENV_DIR

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies using uv
echo "Installing dependencies..."
uv pip install -r requirements.txt

# Install the package in development mode
echo "Installing package in development mode..."
uv pip install -e .

# Install pre-commit hooks if available
if command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit hooks..."
    pre-commit install
else
    echo "pre-commit not found. Skipping hook installation."
    echo "Run 'pip install pre-commit && pre-commit install' to set up hooks later."
fi

echo "Setup complete. The package is now installed in development mode."
echo "This means you can import from 'src' directly in both scripts and tests."
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "You can now run:"
echo "  python -m src train [args]        # Run the training script"
echo "  python -m src evaluate [args]     # Run the evaluation script"
echo "  pytest tests/                     # Run tests"
