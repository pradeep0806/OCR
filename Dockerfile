# Base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install PaddlePaddle (CPU-only) and PaddleOCR
RUN pip install paddlepaddle==2.5.0 paddleocr==2.6.0.3

# Disable AVX instructions if they are causing issues
ENV FLAGS_use_mkldnn=False
ENV FLAGS_use_ngraph=False

# Install other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . /app
WORKDIR /app

# Expose the port your app runs on
EXPOSE 8080

# Run your application with uvicorn
CMD ["uvicorn", "webapp_no_temp:app", "--host", "0.0.0.0", "--port", "8080"]
