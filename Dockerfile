@"
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y cmake build-essential libopenblas-dev liblapack-dev libx11-dev && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
CMD ["python", "app.py"]
"@ | Out-File -FilePath Dockerfile -Encoding utf8
