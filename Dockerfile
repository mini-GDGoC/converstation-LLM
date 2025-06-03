# Dockerfile
FROM python:3.9
WORKDIR /app

# OpenCV 및 Paddle 의존성
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx wget && \
    wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb && \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb || true && \
    apt-get install -f -y && \
    rm libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip /tmp/*

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
