FROM rockylinux:8.7

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies
RUN yum update -y && yum install -y \
    git \
    make \
    gcc \
    gcc-c++ \
    openssl-devel \
    zlib-devel \
    bzip2-devel \
    readline-devel \
    sqlite-devel \
    wget \
    curl \
    llvm \
    ncurses-devel \
    xz \
    xz-devel \
    tk-devel \
    libxml2-devel \
    libxmlsec1-devel \
    libffi-devel \
    liblzma-devel \
    ffmpeg \
    libSM \
    libXext \
    libXrender \
    cmake \
    mesa-libGL \
    python3-devel \
    && yum clean all

# Create user
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

# Copy requirements
COPY --chown=user:user requirements.txt $WORKDIR/requirements.txt

# Install CPU-specific requirements with natten CPU wheel
RUN pip install --no-cache-dir --upgrade -r $WORKDIR/requirements.txt

# Verify natten installation
RUN python -c "import natten; print(f'NATTEN version: {natten.__version__}')"

# Copy application code
COPY --chown=user:user . .

# Set permissions
USER root
RUN chown -R user:user $HOME && \
    chmod -R 755 $HOME && \
    chown -R user:user $WORKDIR && \
    chmod -R 755 $WORKDIR

USER user

# Download example images
RUN mkdir -p examples && \
    wget https://praeclarumjj3.github.io/files/ade20k.jpeg -P $WORKDIR/examples/ && \
    wget https://praeclarumjj3.github.io/files/cityscapes.png -P $WORKDIR/examples/ && \
    wget https://praeclarumjj3.github.io/files/coco.jpeg -P $WORKDIR/examples/

# CPU environment variables
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

EXPOSE 7860
ENTRYPOINT ["python", "gradio_app.py"]
