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

class settings(BaseSettings):
    # Đường dẫn gốc đến thư mục data, sử dụng đường dẫn tuyệt đối và chuẩn hóa cho Windows
    DATA_DIR: str = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "data")))
    MODEL_PATH: str = os.path.normpath(os.path.join(DATA_DIR, "models"))
    FAISS_PATH: str = os.path.normpath(os.path.join(DATA_DIR, "db_faiss"))


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

