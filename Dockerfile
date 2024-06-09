# Use python:3.10-slim as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy requirements.txt to the container
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        libgl1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*  # Remove apt cache after installation to reduce image size

# Install Uvicorn and other Python dependencies
RUN pip install --no-cache-dir uvicorn && \
    pip install --no-cache-dir -r requirements.txt


# Copy your Python application code into the container
COPY . .
COPY models models
# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Command to run Uvicorn when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]