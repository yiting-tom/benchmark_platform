# ===========================================
# CV Benchmark Platform - Dockerfile
# ===========================================
# Multi-stage build for smaller production image

# ---------- Stage 1: Builder ----------
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev


# ---------- Stage 2: Runtime ----------
FROM python:3.12-slim as runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser . .

# Create directories for media and static files
RUN mkdir -p /app/media /app/staticfiles && \
    chown -R appuser:appuser /app/media /app/staticfiles

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Copy startup script
COPY --chown=appuser:appuser scripts/start.sh /app/scripts/start.sh
RUN chmod +x /app/scripts/start.sh

# Default command
CMD ["/app/scripts/start.sh"]
