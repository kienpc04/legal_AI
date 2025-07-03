# # # qa-service/Dockerfile
# # # Sử dụng image Python 3.9 slim để giảm kích thước
# FROM python:3.10-slim

# # Đặt thư mục làm việc
# WORKDIR /app

# # Sao chép file requirements.txt và cài đặt các phụ thuộc
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Sao chép toàn bộ mã nguồn vào container
# COPY . .

# # Mở cổng 8000 (cổng mà FastAPI chạy)
# EXPOSE 8686

# # Lệnh chạy API
# CMD ["python", "app/main.py"]
# 1. Base image: CUDA (nếu dùng GPU) hoặc python slim
# Dockerfile cho qa-service chạy trên Vertex AI
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/build_db.py

ENV PYTHONUNBUFFERED=1

EXPOSE 8686
CMD ["python", "app/main.py"]
