ARG PYTHON_IMAGE_VERSION=3.11
ARG PROJECT_NAME

# BUILDER STAGE #

FROM python:${PYTHON_IMAGE_VERSION}-slim AS builder

# Set environment variables to optimize Python
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends gcc default-libmysqlclient-dev pkg-config

COPY requirements.txt .
COPY docker/app/requirements.txt base.requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -r base.requirements.txt

# FINAL STAGE #

FROM python:${PYTHON_IMAGE_VERSION}-slim

# Declare the ARG and pass it to ENV
ARG PYTHON_IMAGE_VERSION
ENV PYTHON_IMAGE_VERSION=${PYTHON_IMAGE_VERSION}
ARG PROJECT_NAME
ENV PROJECT_NAME=${PROJECT_NAME}

# Set environment variables to optimize Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends tini default-libmysqlclient-dev &&\
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy the Python dependencies from the builder stage
COPY --from=builder /usr/local/lib/python${PYTHON_IMAGE_VERSION}/site-packages/ /usr/local/lib/python${PYTHON_IMAGE_VERSION}/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Set the working directory

RUN useradd -m -r appuser && \
    mkdir /app && \
    chown -R appuser /app

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