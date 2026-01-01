from pydantic import BaseModel
from typing import Dict

# [Request] 사용자 입력
class PredictRequest(BaseModel):
    text: str

# [Response] API 응답
class PredictResponse(BaseModel):
    emotion: str
    emotion_conf: float
    situation: str
    situation_conf: float
    # 필요 시 전체 확률 정보도 포함
    emotion_probs: Dict[str, float] = {}
    situation_probs: Dict[str, float] = {}