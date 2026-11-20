# Multi-stage production Dockerfile for GeneWeaver GPU-Accelerated CRISPR Suite
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src"

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set default python alias
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    streamlit \
    plotly \
    pandas \
    numpy \
    biopython \
    textual \
    rich \
    reportlab \
    pytest

# Copy application source code
COPY . .

# Expose Streamlit web portal port
EXPOSE 8501

# Healthcheck to verify Streamlit server status
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default launch command
ENTRYPOINT ["streamlit", "run", "src/geneweaver/web/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--theme.base=dark"]
