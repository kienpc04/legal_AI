# qa-service/app/main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from config import settings

# Khởi tạo FastAPI
app = FastAPI()

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
vector_store = FAISS.load_local(settings.FAISS_PATH, embeddings=embedder,allow_dangerous_deserialization=True)
llm = HuggingFacePipeline.from_model_id(
    model_id=f"{settings.MODEL_PATH}/vinallama-7b-chat",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 512},
)

# Mô hình dữ liệu
class QARequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    answer: str

# Xác thực JWT
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# Logic hỏi đáp
def get_answer(question: str) -> str:
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Bạn là một trợ lý pháp lý thông minh.Hãy dựa trên các đoạn văn bản luật sau để trả lời câu hỏi của người dùng một cách chính xác và ngắn gọn.\nCâu hỏi: {question}\nThông tin: {context}\nTrả lời:"
    answer = llm(prompt)
    return answer

# API endpoint
@app.post("/api/qa", response_model=QAResponse)
async def qa_endpoint(request: QARequest, _ = Depends(verify_token)):
    try:
        answer = get_answer(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8686)