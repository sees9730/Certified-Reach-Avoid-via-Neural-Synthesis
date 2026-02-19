# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for scientific computing and git
RUN apt-get update && apt-get install -y \
    git \
    graphviz

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Note: auto_LiRPA requires git to install from the GitHub repository
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and examples
COPY src/ ./src/
COPY examples/ ./examples/
COPY README.md .

# Set environment variables for PyTorch
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command - show help
CMD ["python", "-c", "print('Neural Certificate Docker Container\\n\\nTo run an example:\\n  docker run -v $(pwd):/app neural-certificate python examples/synthesis/<example>/main.py --train=1')"]
