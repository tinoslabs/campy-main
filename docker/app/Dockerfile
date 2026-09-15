ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=13.1.1
ARG UBUNTU_VERSION=24.04
ARG PROJECT_NAME

# BUILDER STAGE #

FROM python:${PYTHON_VERSION}-slim AS builder

# Set environment variables to optimize Python
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends gcc g++ default-libmysqlclient-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY docker/app/requirements.txt base.requirements.txt

RUN pip install --upgrade pip wheel && \
    pip wheel --no-cache-dir \
        -r requirements.txt \
        -r base.requirements.txt \
        -w /wheels

# RUNTIME STAGE #

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Declare the ARG and pass it to ENV
ARG PYTHON_VERSION
ARG PROJECT_NAME
ENV PROJECT_NAME=${PROJECT_NAME}

# Set environment variables to optimize Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python3-pip \
        tini default-libmysqlclient-dev libgl1 libglib2.0-0 && \
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python dependencies from the builder stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --break-system-packages /wheels/* && rm -rf /wheels

# Set the working directory

RUN useradd -m -r appuser && \
    mkdir /app && \
    chown -R appuser:appuser /app

WORKDIR /app

RUN mkdir -p staticfiles mediafiles && \
    chown -R appuser:appuser staticfiles mediafiles

# Copy application code
COPY --chown=appuser:appuser . .

COPY --chown=appuser:appuser docker/app/prod_settings.py /app/${PROJECT_NAME}/prod_settings.py

# Copy the entrypoint script
COPY --chown=appuser:appuser docker/app/start.sh start.sh
RUN chmod +x start.sh

# Switch to non-root user
USER appuser

# Expose the application port
EXPOSE 8000 

ENTRYPOINT ["tini", "-g", "--"]

CMD ["/bin/sh", "start.sh"]
