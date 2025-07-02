import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Đường dẫn
DATA_DIR = "app/data/processed"
FAISS_PATH = "app/data/db_faiss"

# Load mô hình embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_vector_store():
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(file_path) or not filename.endswith(".txt"):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"source": filename}))

    if not documents:
        print("Không có văn bản nào để xử lý.")
        return

    print(f"Đang xây dựng vector store từ {len(documents)} đoạn văn bản...")

    # Backup FAISS cũ nếu có
    if os.path.exists(FAISS_PATH):
        backup_path = FAISS_PATH + "_backup"
        os.rename(FAISS_PATH, backup_path)
        print(f"Đã backup FAISS cũ sang: {backup_path}")

    # Tạo vector store và lưu
    vector_store = FAISS.from_documents(documents, embedder)
    vector_store.save_local(FAISS_PATH)
    print(f"Đã lưu vector store vào: {FAISS_PATH}")

if __name__ == "__main__":
    build_vector_store()
