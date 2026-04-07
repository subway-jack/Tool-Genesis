FROM python:3.13-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .
ENV PYTHONPATH=/app

# Usage
CMD ["python", "-c", "print('Tool-Genesis Benchmark\\n\\nUsage:\\n  # Generate:\\n  python scripts/run_benchmark/generate_mcp_from_task.py --help\\n\\n  # Evaluate:\\n  python scripts/run_benchmark/run_evaluation.py --help')"]
