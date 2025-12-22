# ==========================================
# 02_preprocessing.py
# ==========================================
import os
import re
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


# ==========================================
# 0. 환경 및 경로 설정
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/raw")
SAVE_PATH = os.path.join(BASE_DIR, "../data/processed")

TRAIN_FILE = os.path.join(DATA_PATH, "감성대화말뭉치_Training.xlsx")
VALID_FILE = os.path.join(DATA_PATH, "감성대화말뭉치_Validation.xlsx")

# ====================
# 1. Text Cleaning
# ====================
def clean_text(text):
    """공백/개행 제거 + 문자열 정규화"""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()

# ====================
# 2. 문장 병합
# ====================
def merge_sentences(df):
    """사람문장1~3을 안전하게 병합"""
    sentence_cols = [c for c in ["사람문장1", "사람문장2", "사람문장3"] if c in df.columns]
    df[sentence_cols] = df[sentence_cols].fillna("").astype(str)
    df["merged_text"] = df[sentence_cols].agg(" ".join, axis=1).apply(clean_text)

    # 완전히 빈 텍스트 제거
    df = df[df["merged_text"] != ""]
    return df

# ==============================
# 3. Multi-task Label Encoder
# ==============================
def encode_labels(train_df, valid_df, save_path):
    """감정 + 상황 인코딩"""
    # Emotion
    emo_labels = sorted(train_df["감정_대분류"].unique())
    emo2id = {label: idx for idx, label in enumerate(emo_labels)}

    # Situation
    sit_labels = sorted(train_df["상황키워드"].unique())
    sit2id = {label: idx for idx, label in enumerate(sit_labels)}

    # Apply
    for df in (train_df, valid_df):
        df["label_emotion"] = df["감정_대분류"].map(emo2id)
        df["label_situation"] = df["상황키워드"].map(sit2id)

    # 저장 (디렉토리 생성은 save_csv 등 외부에서 보장되거나 여기서 생성)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    label_map = {
        "emotion2id": emo2id,
        "situation2id": sit2id,
        "id2emotion": {v: k for k, v in emo2id.items()},
        "id2situation": {v: k for k, v in sit2id.items()}
    }
    
    with open(os.path.join(save_path, "label_map.pkl"), "wb") as f:
        pickle.dump(label_map, f)

    print(f"[INFO] Emotion Classes: {len(emo_labels)}, Situation Classes: {len(sit_labels)}")

    return train_df, valid_df

# ===============
# 5. Save CSV
# ===============
def save_csv(train_df, valid_df, save_path):
    os.makedirs(save_path, exist_ok=True)

    train_df.to_csv(os.path.join(save_path, "train.csv"), index=False, encoding="utf-8-sig")
    valid_df.to_csv(os.path.join(save_path, "valid.csv"), index=False, encoding="utf-8-sig")

    print(f"[INFO] CSV Saved → {save_path}")

# =====================
# 6. Main Pipeline
# =====================
def run_preprocessing_multitask(
    train_path,
    valid_path,
    save_path
):
    print("\n===== Multi-Task Preprocessing Start =====")

    # Load & Merge
    # 로컬 경로 문제 시 예외 처리를 위해 try-except 블록이나 경로 확인 로직 추가 가능하지만
    # 요청대로 메인 코드는 최대한 유지함
    print(f"Loading Train Data: {train_path}")
    train_df = merge_sentences(pd.read_excel(train_path))
    
    print(f"Loading Valid Data: {valid_path}")
    valid_df = merge_sentences(pd.read_excel(valid_path))

    # Encode Labels
    train_df, valid_df = encode_labels(train_df, valid_df, save_path)

    # Save CSV
    save_csv(train_df, valid_df, save_path)

    print("===== Preprocessing Completed =====\n")
    return train_df, valid_df

# =====================
# Execution Block
# =====================
if __name__ == "__main__":
    if os.path.exists(TRAIN_FILE) and os.path.exists(VALID_FILE):
        run_preprocessing_multitask(
            train_path=TRAIN_FILE,
            valid_path=VALID_FILE,
            save_path=SAVE_PATH
        )
    else:
        print(f"[ERROR] 데이터 파일을 찾을 수 없습니다.")
        print(f"확인된 경로: {DATA_PATH}")
        print("data 폴더 안에 엑셀 파일이 있는지 확인해주세요.")