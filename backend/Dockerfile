FROM python:3.12.10-slim

WORKDIR /app

# Install system dependencies for data science/ML packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Demo deployment: work orders are approved without an approver token (see _authorise in main.py)
ENV APPROVAL_MODE=open
# Approved work orders (/app/maintenance_history_db) and the audit log (/app/persist) must survive
# redeploys: mount both paths as CapRover persistent directories.
ENV AUDIT_LOG_PATH=/app/persist/audit_log.jsonl
RUN mkdir -p /app/persist /app/maintenance_history_db

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
