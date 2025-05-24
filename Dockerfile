FROM python:3.10

WORKDIR /home/user/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget build-essential \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 ffmpeg cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy and run installation script
COPY install_deps.sh .
RUN chmod +x install_deps.sh && ./install_deps.sh

# Copy application
COPY . /home/user/app

USER user

ENV PYTHONPATH=/home/user/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
