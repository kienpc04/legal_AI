# # qa-service/app/config.py
# from pydantic_settings import BaseSettings

# class Settings(BaseSettings):
#     MODEL_PATH: str = "/app/data/models"
#     FAISS_PATH: str = "/app/data/db_faiss"
#     JWT_SECRET: str = "your_jwt_secret"
#     JWT_ALGORITHM: str = "HS256"

#     class Config:
#         env_file = ".env"
#         env_file_encoding = "utf-8"

# settings = Settings()
import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Đường dẫn gốc đến thư mục data, sử dụng đường dẫn tuyệt đối và chuẩn hóa cho Windows
    MODEL_PATH: str = Field(..., env="MODEL_PATH")
    FAISS_PATH: str = Field(..., env="FAISS_PATH")
    JWT_SECRET: str = Field(..., env="JWT_SECRET")


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
settings = Settings()
