# 任意：Podman で API を動かす
#   podman build -t nikkei-forecast -f Containerfile .
#   podman run --rm -p 127.0.0.1:8000:8000 \
#     -v ./data:/app/data -v ./models:/app/models -v ./predictions:/app/predictions \
#     nikkei-forecast
# data/・models/・predictions/ はボリュームでマウントする（イメージには入れない）
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config.yaml .
COPY src ./src
COPY reports/lgb_best_params.json ./reports/
ENV PYTHONIOENCODING=utf-8 TZ=Asia/Tokyo
EXPOSE 8000
# コンテナ内では 0.0.0.0 で待ち受け、公開範囲はホスト側の -p 127.0.0.1:8000:8000 で絞る
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
