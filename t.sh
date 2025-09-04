#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run the application
echo "Starting NeuroNest application at $(date)"
python app.py

echo "Job completed at $(date)"
