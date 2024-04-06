# Step 1: Use the official Python 3 base image from Docker Hub
FROM python:3.9-slim as builder

# Set environment variables to reduce Python package issues and ensure output is sent straight to the terminal without buffering
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies required for Python packages (if any)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies in .venv
COPY requirements.txt .
RUN python -m venv .venv
RUN . .venv/bin/activate && pip install -r requirements.txt

# Step 2: Build a smaller image from the builder stage
FROM python:3.9-slim as final

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv .venv

# Ensure scripts in .venv are usable:
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application
COPY . .

# Create a non-root user to run the application
RUN useradd --create-home appuser
USER appuser

# Inform Docker that the container listens on port 5000
EXPOSE 5001

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "aiapp:app"]