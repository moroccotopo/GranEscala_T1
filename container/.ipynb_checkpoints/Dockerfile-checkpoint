FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    ca-certificates \
    curl \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv
RUN pip install --no-cache-dir flask gunicorn joblib numpy pandas scikit-learn

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PYTHONPATH=/opt/program
ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program

COPY pyproject.toml uv.lock /opt/program/
RUN uv sync --frozen --no-dev || true

COPY src /opt/program/src
COPY container/ridge_sales /opt/program
COPY container/ridge_sales/nginx.conf /etc/nginx/nginx.conf

RUN chmod +x /opt/program/train /opt/program/serve

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["if [ \"$1\" = 'train' ]; then shift; exec /opt/program/train \"$@\"; else exec /opt/program/serve; fi", "--"]