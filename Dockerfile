# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# ffmpeg is likely needed for audio processing
RUN apt-get update && apt-get install -y \
    # ffmpeg \
    # gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the project configuration files
COPY pyproject.toml .

# Install Python dependencies
# We install the current directory in editable mode or just the dependencies
# Using pip to install dependencies defined in pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
