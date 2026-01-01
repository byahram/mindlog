import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import gradio as gr
from transformers import AutoTokenizer, AutoModel

def set_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans KR"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return "sans-serif"

set_korean_font()


# =========================
# 0. Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    MODEL_NAME = "beomi/KcELECTRA-base-v2022"
    CKPT_NAME  = "best_multitask_model.bin"

    MAX_LEN = 128
    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(Config.MODEL_DIR, exist_ok=True)
print("[INFO] Device:", Config.DEVICE)


# =========================
# 1. Load label_map
# =========================
label_map_path = os.path.join(Config.DATA_DIR, "label_map.pkl")
if not os.path.exists(label_map_path):
    raise FileNotFoundError(f"label_map.pkl not found: {label_map_path}")

with open(label_map_path, "rb") as f:
    label_map = pickle.load(f)

id2emotion   = label_map["id2emotion"]
id2situation = label_map["id2situation"]

num_emo = len(label_map["emotion2id"])
num_sit = len(label_map["situation2id"])

emo_names = [id2emotion[i] for i in range(num_emo)]
sit_names = [id2situation[i] for i in range(num_sit)]

print("[INFO] num_emo:", num_emo, "num_sit:", num_sit)


# =========================
# 2. Model + tokenizer + ckpt load
# =========================
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

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
model = SentimentMultiTaskModel(Config.MODEL_NAME, num_emo, num_sit).to(Config.DEVICE)

# 체크포인트 로드
ckpt_path = os.path.join(Config.MODEL_DIR, Config.CKPT_NAME)
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location=Config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    print("[INFO] Loaded ckpt:", ckpt_path)
else:
    print("[WARN] Checkpoint not found. Running with random weights.")


# =========================
# 3. Predict / history / save
# =========================
def predict_probs(text: str):
    """모델 예측 후 dict 형태로 반환 (Gradio Label용)"""
    text = (text or "").strip()
    enc = tokenizer(text, max_length=Config.MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    enc = {k: v.to(Config.DEVICE) for k, v in enc.items()}

    model.eval()
    with torch.inference_mode():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    emo_probs_t = F.softmax(out["logits_emotion"], dim=-1)[0]
    sit_probs_t = F.softmax(out["logits_situation"], dim=-1)[0]

    emo_dict = {emo_names[i]: float(emo_probs_t[i]) for i in range(len(emo_names))}
    sit_dict = {sit_names[i]: float(sit_probs_t[i]) for i in range(len(sit_names))}

    # Best Picking
    emo_id = int(torch.argmax(emo_probs_t).item())
    sit_id = int(torch.argmax(sit_probs_t).item())
    
    return emo_id, emo_probs_t.max().item(), sit_id, sit_probs_t.max().item(), emo_dict, sit_dict

def append_turn(history, user_text, emo_id, emo_conf, sit_id, sit_conf):
    turn = len(history) + 1
    history.append({
        "turn": turn,
        "ts": datetime.now().strftime("%H:%M:%S"), 
        "text": user_text,
        "emotion": id2emotion[emo_id],
        "emo_conf": round(emo_conf, 2),
        "situation": id2situation[sit_id],
        "sit_conf": round(sit_conf, 2),
        "emo_id": emo_id,
        "sit_id": sit_id 
    })
    return history

# def save_history_csv(history, filename_prefix="chat_history"):
#     ts = datetime.now().strftime("%y%m%d_%H%M")
#     filename = f"{filename_prefix}_{ts}.csv"
#     path = os.path.join(Config.MODEL_DIR, filename)

#     df = pd.DataFrame(history).drop(columns=["emotion_probs", "situation_probs"], errors="ignore")
#     df.to_csv(path, index=False, encoding="utf-8-sig")
#     return path

# def format_current_pred(emo_id, emo_conf, sit_id, sit_conf):
#     emo = id2emotion[emo_id]
#     sit = id2situation[sit_id]
#     return (
#         f"### 현재 예측\n"
#         f"- 감정: **{emo}** (conf={emo_conf:.2f})\n"
#         f"- 상황: **{sit}** (conf={sit_conf:.2f})\n\n"
#         f"**confidence** = 모델이 고른 1등 라벨의 확률(softmax)."
#     )

# =========================
# 4. Plot Functions (Timeline)
# =========================
def draw_timeline(history):
    if not history:
        return None, None
    
    df = pd.DataFrame(history)
    
    # Emotion Timeline
    fig_emo = plt.figure(figsize=(12, 4))
    # 점 크기를 confidence에 비례하게 (최소 50, 최대 300)
    sizes = df["emo_conf"] * 300 
    plt.scatter(df["turn"], df["emo_id"], s=sizes, c=df["emo_id"], cmap="tab10", alpha=0.7)
    plt.yticks(range(len(emo_names)), emo_names)
    plt.xlabel("Turn")
    plt.title("Emotion Flow")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Situation Timeline
    fig_sit = plt.figure(figsize=(12, 4))
    sizes_sit = df["sit_conf"] * 300
    plt.scatter(df["turn"], df["sit_id"], s=sizes_sit, c=df["sit_id"], cmap="Set2", alpha=0.7)
    plt.yticks(range(len(sit_names)), sit_names)
    plt.xlabel("Turn")
    plt.title("Situation Flow")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    return fig_emo, fig_sit

# =========================
# 5. UI Event Handlers
# =========================
def on_submit(text, history):
    if not text.strip():
        return "", history, None, None, None, pd.DataFrame()

    # 1. Predict
    e_id, e_conf, s_id, s_conf, e_dict, s_dict = predict_probs(text)
    
    # 2. Update History
    new_history = append_turn(history, text, e_id, e_conf, s_id, s_conf)
    
    # 3. Create Charts
    fig_emo, fig_sit = draw_timeline(new_history)
    df = pd.DataFrame(new_history).drop(columns=["emo_id", "sit_id"]) # 보여주기용 DF에서는 ID 제외
    
    # 반환: 입력창초기화, 히스토리업데이트, 감정라벨, 상황라벨, 감정차트, 상황차트, 데이터프레임
    return "", new_history, e_dict, s_dict, fig_emo, fig_sit, df

def on_reset():
    return [], None, None, None, None, pd.DataFrame()

# =========================
# 6. Gradio Build (Clean Version)
# =========================
def build_app():
    with gr.Blocks() as demo:
        
        # [Header]
        gr.Markdown(
            """
            # 🎋 Mind Log: Silent Tracker
            ### "말하지 않아도 드러나는 감정의 흐름"
            """
        )
        
        # State (전역 변수 대신 세션별 저장소)
        state_history = gr.State([])

        # [Tabs] 기능을 탭으로 분리하여 깔끔하게
        with gr.Tabs():
            
            # --- TAB 1: 기록 및 즉시 분석 (Main) ---
            with gr.TabItem("📝 기록하기 (Record)"):
                
                # [상단] 입력 영역 (전체 너비 사용)
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(
                            label="지금 어떤 마음인가요?", 
                            placeholder="자유롭게 털어놓으세요 (Enter로 입력)", 
                            lines=5
                        )
                        btn_submit = gr.Button("기록하기", variant="primary")
                
                # 디자인적 구분을 위한 여백 및 헤더
                gr.Markdown("---") 
                gr.Markdown("### 🔍 분석 결과")

                # [하단] 분석 결과 영역 (감정과 상황을 나란히 배치)
                with gr.Row():
                    with gr.Column(scale=1):
                        out_emo_label = gr.Label(label="감정 (Emotion)", num_top_classes=3)
                    
                    with gr.Column(scale=1):
                        out_sit_label = gr.Label(label="상황 (Situation)", num_top_classes=3)

            # --- TAB 2: 대시보드 (Dashboard) ---
            with gr.TabItem("📊 대시보드 (Dashboard)"):
                gr.Markdown("### 🌊 감정과 상황의 흐름")
                plot_emo = gr.Plot(label="Emotion Timeline")
                plot_sit = gr.Plot(label="Situation Timeline")
                
            # --- TAB 3: 데이터 로그 (Data Log) ---
            with gr.TabItem("📂 데이터 (History)"):
                with gr.Row():
                    btn_reset = gr.Button("🗑️ 기록 초기화", size="sm", variant="stop")
                    # btn_save = gr.Button("💾 CSV 저장", size="sm") # (추후 구현)
                history_table = gr.Dataframe(
                    headers=["turn", "ts", "text", "emotion", "emo_conf", "situation", "sit_conf"],
                    datatype=["number", "str", "str", "str", "number", "str", "number"],
                    interactive=False
                )

        # [Event Linking]
        # 1. Submit (Enter or Click)
        # 출력 순서: [입력창, state, 감정라벨, 상황라벨, 감정차트, 상황차트, 테이블]
        input_text.submit(
            fn=on_submit,
            inputs=[input_text, state_history],
            outputs=[input_text, state_history, out_emo_label, out_sit_label, plot_emo, plot_sit, history_table]
        )
        btn_submit.click(
            fn=on_submit,
            inputs=[input_text, state_history],
            outputs=[input_text, state_history, out_emo_label, out_sit_label, plot_emo, plot_sit, history_table]
        )
        
        # 2. Reset
        btn_reset.click(
            fn=on_reset,
            inputs=[],
            outputs=[state_history, out_emo_label, out_sit_label, plot_emo, plot_sit, history_table]
        )

    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860)