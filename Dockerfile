FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir setuptools
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

COPY . .

RUN mkdir -p uploads outputs

CMD ["python", "main.py"]
