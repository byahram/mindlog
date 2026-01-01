import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from api.schemas import PredictRequest, PredictResponse
from api.inference import MindLogHandler

# 전역 핸들러 변수
handler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global handler
    print("[INFO] Loading MindLog Model...")
    # 핸들러 초기화 (Config 경로는 api/config.py에서 알아서 처리함)
    handler = MindLogHandler()
    yield
    print("[INFO] Shutting down...")

app = FastAPI(
    title="MindLog API",
    version="1.0",
    description="MindLog 감정/상황 분석 모델 서빙 API",
    lifespan=lifespan
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "MindLog API is running."}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not handler:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    
    result = handler.predict(req.text)
    
    if result is None:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    return result

if __name__ == "__main__":
    # Root에서 실행하므로 "main:app"으로 실행
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)