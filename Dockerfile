FROM python:3.11-slim

# opencv-python requires these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

# Dataset is mounted at runtime — never baked into the image
ENV VINDR_ROOT=/data/vindr-mammo

ENTRYPOINT ["python"]
CMD ["src/compute_iqms_full_dataset.py", "--output-dir", "/app/results"]
