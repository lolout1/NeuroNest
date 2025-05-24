FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget build-essential libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and run installation script
COPY install_deps.sh .
RUN chmod +x install_deps.sh && ./install_deps.sh

# Copy application
COPY . /app

# Setup user and environment
RUN useradd -m neuronest && chown -R neuronest:neuronest /app
USER neuronest

ENV PYTHONPATH=/app GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860

EXPOSE 7860
CMD ["python", "app.py"]
