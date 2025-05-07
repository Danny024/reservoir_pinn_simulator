# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Set non-interactive frontend to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create directories for MLflow and artifacts
RUN mkdir -p /app/mlruns /app/artifacts

# Expose ports for Streamlit, FastAPI, MLflow, and Ollama
EXPOSE 8501 8000 5000 11434

# Create a startup script
RUN echo '#!/bin/bash\n\
# Start Ollama server in the background\n\
ollama serve &\n\
sleep 5\n\
# Pull DeepSeek model\n\
ollama pull deepseek-r1:1.5b\n\
# Start MLflow server in the background\n\
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:///app/mlruns --default-artifact-root file:///app/artifacts &\n\
# Start FastAPI server in the background\n\
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 &\n\
# Start Streamlit app\n\
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0\n\
' > start.sh && chmod +x start.sh

# Run the startup script
CMD ["./start.sh"]