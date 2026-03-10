#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${1:-ridge-sales}
REGION=${AWS_DEFAULT_REGION:-us-east-1}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

FULLNAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:latest"

chmod +x container/ridge_sales/train
chmod +x container/ridge_sales/serve

aws ecr describe-repositories --repository-names "${IMAGE_NAME}" --region "${REGION}" >/dev/null 2>&1 || \
aws ecr create-repository --repository-name "${IMAGE_NAME}" --region "${REGION}" >/dev/null

aws ecr get-login-password --region "${REGION}" \
| docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

docker build -f container/Dockerfile -t "${IMAGE_NAME}:latest" .
docker tag "${IMAGE_NAME}:latest" "${FULLNAME}"
docker push "${FULLNAME}"

echo "Imagen subida a: ${FULLNAME}"