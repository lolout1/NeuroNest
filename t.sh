#!/bin/bash

# Source .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Loaded environment from .env"
fi

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run the application
echo "Starting NeuroNest application at $(date)"
python app.py

echo "Job completed at $(date)"
