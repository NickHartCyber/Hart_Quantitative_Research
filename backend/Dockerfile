# Backend container for the Hart Quantitative Research FastAPI service
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Create writable directories that are bind-mounted in docker-compose
RUN mkdir -p /app/files/data /app/files/logs /app/files/daily_suggestions

# System deps for numpy/pandas + build ta-lib from source (not available in base repos)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    chromium \
    chromium-driver \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libgbm1 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    wget \
    tzdata \
    ca-certificates \
    && \
    mkdir -p /tmp/ta-lib && \
    wget -q -O /tmp/ta-lib/ta-lib-0.4.0-src.tar.gz https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf /tmp/ta-lib/ta-lib-0.4.0-src.tar.gz -C /tmp/ta-lib --strip-components=1 && \
    cd /tmp/ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    rm -rf /tmp/ta-lib && \
    apt-get purge -y wget && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml pyproject.additions.toml README.md ./
COPY backend ./backend
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e ".[politician-trades-scrapers]"

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
