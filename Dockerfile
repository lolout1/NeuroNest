FROM python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU version first
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Clone and install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 && \
    pip install -e detectron2

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app files
COPY . .

# Run the app
CMD ["python", "app.py"]
