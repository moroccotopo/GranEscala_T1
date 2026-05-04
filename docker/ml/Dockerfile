FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/program
WORKDIR /opt/program
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY docker/ml/requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /opt/program/src/
COPY deployment/ /opt/program/deployment/
ENTRYPOINT ["python", "-m", "src.cloud_pipeline.run_step"]
