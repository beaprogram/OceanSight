FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285
WORKDIR /app
COPY requirements-inference-lock.txt .
RUN pip install --no-cache-dir -r requirements-inference-lock.txt && useradd --uid 10001 --create-home oceansight
COPY oceansight/ ./oceansight/
ENV OCEANSIGHT_MODEL=/models/best.onnx
USER oceansight
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"
CMD ["uvicorn", "oceansight.api:app", "--host", "0.0.0.0", "--port", "8000"]
