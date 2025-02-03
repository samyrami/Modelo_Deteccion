FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p ~/.streamlit && \
    chmod -R 777 ~/.streamlit

# Environment variables
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

# Create a startup script
RUN echo '#!/bin/bash\n\
export PORT="${PORT:-8501}"\n\
export PYTHONPATH="/app:${PYTHONPATH}"\n\
export STREAMLIT_SERVER_PORT="$PORT"\n\
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"\n\
\n\
# Start Streamlit\n\
exec streamlit run app.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Add these environment variables in your Dockerfile
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
# Expose port
EXPOSE 8501

# Health check with longer interval and timeout
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
  CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

CMD ["/app/start.sh"]