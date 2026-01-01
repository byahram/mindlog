import os
import torch

# 현재 파일(api/config.py)의 부모의 부모 폴더가 프로젝트 루트(MindLog/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    MODEL_NAME = "beomi/KcELECTRA-base-v2022"
    CKPT_NAME  = "best_multitask_model.bin"
    LABEL_MAP_NAME = "label_map.pkl"

    MAX_LEN = 128
    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 파일 경로 확인용
    CKPT_PATH = os.path.join(MODEL_DIR, CKPT_NAME)
    LABEL_MAP_PATH = os.path.join(DATA_DIR, LABEL_MAP_NAME)