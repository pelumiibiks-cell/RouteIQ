FROM python:3.11-slim

# Lambda Web Adapter: translates Lambda invocations into plain HTTP against the
# uvicorn process below, so this stays an ordinary web-server image that also
# runs on App Runner or Fargate unchanged.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config
COPY frontend ./frontend

ENV PYTHONPATH=/app
ENV PORT=8000
# "/" redirects to /dashboard, which the adapter's readiness check won't accept
ENV AWS_LWA_READINESS_CHECK_PATH=/api

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
