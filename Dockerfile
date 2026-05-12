# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install system-level audio dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 2. Environment Fix: Pin setuptools to keep 'pkg_resources' active
# Also ensures 'av' is installed as a binary to avoid the GCC compiler trap.
RUN pip install --no-cache-dir --upgrade pip "setuptools<82.0.0" wheel && \
    pip install --no-cache-dir --only-binary=:all: av==12.1.0 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "app.main"]