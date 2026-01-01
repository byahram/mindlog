import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from api.config import Config 

# 1. 모델 아키텍처 정의
class SentimentMultiTaskModel(nn.Module):
    def __init__(self, model_name, num_emo_classes, num_sit_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.emo_classifier = nn.Linear(hidden, num_emo_classes)
        self.sit_classifier = nn.Linear(hidden, num_sit_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return {
            "logits_emotion": self.emo_classifier(cls),
            "logits_situation": self.sit_classifier(cls),
        }

# 2. 추론 핸들러 (모델 로드 및 예측 담당)
class MindLogHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.id2emotion = {}
        self.id2situation = {}
        self.emo_names = []
        self.sit_names = []
        
        print(f"[INFO] Initializing Handler on {Config.DEVICE}...")
        self._load_label_map()
        self._load_model()

    def _load_label_map(self):
        if not os.path.exists(Config.LABEL_MAP_PATH):
            raise FileNotFoundError(f"Label map not found at {Config.LABEL_MAP_PATH}")

        with open(Config.LABEL_MAP_PATH, "rb") as f:
            label_map = pickle.load(f)

        self.id2emotion = label_map["id2emotion"]
        self.id2situation = label_map["id2situation"]
        
        self.num_emo = len(label_map["emotion2id"])
        self.num_sit = len(label_map["situation2id"])
        
        self.emo_names = [self.id2emotion[i] for i in range(self.num_emo)]
        self.sit_names = [self.id2situation[i] for i in range(self.num_sit)]

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = SentimentMultiTaskModel(Config.MODEL_NAME, self.num_emo, self.num_sit).to(Config.DEVICE)
        
        if os.path.exists(Config.CKPT_PATH):
            state = torch.load(Config.CKPT_PATH, map_location=Config.DEVICE)
            self.model.load_state_dict(state)
            self.model.eval()
            print(f"[INFO] Model loaded successfully from {Config.CKPT_PATH}")
        else:
            print(f"[WARN] Checkpoint not found at {Config.CKPT_PATH}. Using random weights.")

    def predict(self, text: str):
        if not text or not text.strip():
            return None

        # 전처리
        enc = self.tokenizer(
            text, 
            max_length=Config.MAX_LEN, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        enc = {k: v.to(Config.DEVICE) for k, v in enc.items()}

        # 추론
        with torch.inference_mode():
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

        emo_probs_t = F.softmax(out["logits_emotion"], dim=-1)[0]
        sit_probs_t = F.softmax(out["logits_situation"], dim=-1)[0]

        # 결과 추출
        emo_id = int(torch.argmax(emo_probs_t).item())
        sit_id = int(torch.argmax(sit_probs_t).item())
        
        return {
            "emotion": self.id2emotion[emo_id],
            "emotion_conf": float(emo_probs_t.max().item()),
            "situation": self.id2situation[sit_id],
            "situation_conf": float(sit_probs_t.max().item()),
            "emotion_probs": {self.emo_names[i]: float(emo_probs_t[i]) for i in range(len(self.emo_names))},
            "situation_probs": {self.sit_names[i]: float(sit_probs_t[i]) for i in range(len(self.sit_names))}
        }

import os 