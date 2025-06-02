FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    ffmpeg libsm6 libxext6 cmake libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create user (required by HF Spaces)
RUN useradd -ms /bin/bash user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install pyenv
RUN curl https://pyenv.run | bash
ENV PATH=$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH

# Install Python 3.8.15
RUN pyenv install 3.8.15 && \
    pyenv global 3.8.15 && \
    pyenv rehash && \
    pip install --no-cache-dir --upgrade pip==22.3.1 setuptools wheel

ENV WORKDIR=/code
WORKDIR $WORKDIR

# Switch to root to set permissions
USER root
RUN chown -R user:user $WORKDIR && \
    chmod -R 755 $WORKDIR
USER user

# Copy requirements first for better caching
COPY --chown=user:user requirements.txt $WORKDIR/requirements.txt

# Install CPU requirements
RUN pip install --no-cache-dir --upgrade -r $WORKDIR/requirements.txt

# Copy application code
COPY --chown=user:user . .

# Create examples directory and download sample images
RUN mkdir -p examples && \
    wget -q https://praeclarumjj3.github.io/files/ade20k.jpeg -P $WORKDIR/examples/ && \
    wget -q https://praeclarumjj3.github.io/files/cityscapes.png -P $WORKDIR/examples/ && \
    wget -q https://praeclarumjj3.github.io/files/coco.jpeg -P $WORKDIR/examples/

# Set environment variables for CPU
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

EXPOSE 7860
ENTRYPOINT ["python", "gradio_test.py"]
