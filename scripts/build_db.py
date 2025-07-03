import os
import shutil
import logging
import torch
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import gc

# Configuration
CHUNK_SIZE = 10000
BATCH_SIZE = 256
DATASET_NAME = "sontungkieu/ThuVienPhapLuat"
FAISS_DIR = Path("app/data/faiss_parts")
FINAL_FAISS = Path("app/data/db_faiss")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOTAL_DOCS = 222377  # Total documents from previous log
CACHE_DIR = "D:/hf_cache"  # Use a drive with sufficient space

# Logging
logging.basicConfig(level=logging.INFO, filename="vector_store.log", format="%(asctime)s - %(levelname)s - %(message)s")

# Setup
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Suppress symlink warning
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": device})
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
FAISS_DIR.mkdir(parents=True, exist_ok=True)

def split_and_embed_part(examples, part_idx):
    try:
        documents = []
        for example in tqdm(examples, desc=f"📄 [Part {part_idx}] Splitting", total=len(examples)):
            text = str(example.get("noi_dung", "")).strip()
            if not text:
                logging.warning(f"Empty content in document ID: {example.get('_id', 'unknown')}")
                continue
            chunks = text_splitter.split_text(text)
            metadata = {
                "source": example.get("link", ""),
                "title": example.get("loai_van_ban", ""),
                "id": example.get("_id", "")
            }
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata=metadata))

        logging.info(f"[Part {part_idx}] Generated {len(documents)} chunks")
        print(f"🔹 [Part {part_idx}] Total chunks: {len(documents)}. Encoding...")

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        faiss_store = None

        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"🧠 [Part {part_idx}] Embedding batch"):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_metas = metadatas[i:i + BATCH_SIZE]
            batch_docs = [Document(page_content=t, metadata=m) for t, m in zip(batch_texts, batch_metas)]
            if faiss_store is None:
                faiss_store = FAISS.from_documents(batch_docs, embedder)
            else:
                faiss_store.add_documents(batch_docs)

        path = FAISS_DIR / f"faiss_part_{part_idx}"
        faiss_store.save_local(str(path))
        logging.info(f"[Part {part_idx}] Saved FAISS at: {path}")
        print(f"✅ [Part {part_idx}] Saved FAISS at: {path}")
        del documents, faiss_store, texts, metadatas
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()  # Free GPU memory
    except Exception as e:
        logging.error(f"[Part {part_idx}] Failed: {e}")
        print(f"❌ [Part {part_idx}] Error: {e}")
        raise

def build_vector_store_from_hf():
    try:
        print("📥 Loading dataset from Hugging Face in streaming mode...")
        logging.info("Loading dataset in streaming mode")
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True, cache_dir=CACHE_DIR)

        part_data = []
        part_idx = 0
        for i, example in enumerate(tqdm(dataset, desc="🚀 Processing documents", total=TOTAL_DOCS)):
            part_data.append(example)
            if len(part_data) >= CHUNK_SIZE or i == TOTAL_DOCS - 1:
                with ThreadPoolExecutor(max_workers=1) as executor:  # Single worker to avoid memory issues
                    executor.submit(split_and_embed_part, part_data, part_idx).result()
                part_data = []
                part_idx += 1

        print("🔀 Merging FAISS parts...")
        logging.info("Merging FAISS parts")
        faiss_parts = sorted(FAISS_DIR.glob("faiss_part_*"))
        merged = None
        for part_path in tqdm(faiss_parts, desc="🧩 Merging FAISS"):
            store = FAISS.load_local(str(part_path), embedder, allow_dangerous_deserialization=True)
            if merged is None:
                merged = store
            else:
                merged.merge_from(store)
            del store
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        if FINAL_FAISS.exists():
            shutil.move(str(FINAL_FAISS), str(FINAL_FAISS) + "_backup")
            logging.info(f"Backed up existing FAISS to {FINAL_FAISS}_backup")
            print(f"🔄 Backed up existing FAISS to {FINAL_FAISS}_backup")

        merged.save_local(str(FINAL_FAISS))
        print(f"🎉 Saved final vector store at: {FINAL_FAISS}")
        logging.info(f"Saved final vector store at: {FINAL_FAISS}")

        # Clean up temporary files
        for part_path in faiss_parts:
            shutil.rmtree(part_path)
            logging.info(f"Deleted temporary FAISS: {part_path}")
            print(f"🧹 Deleted temporary FAISS: {part_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Failed: {e}")
        raise

if __name__ == "__main__":
    # Check disk space
    total, used, free = shutil.disk_usage(CACHE_DIR)
    if free < 20 * 1024 * 1024 * 1024:  # Less than 20GB free
        print(f"⚠️ Warning: Only {free / (1024**3):.2f}GB free on disk. Consider freeing space or changing CACHE_DIR.")
        logging.warning(f"Low disk space: {free / (1024**3):.2f}GB free")
    build_vector_store_from_hf()