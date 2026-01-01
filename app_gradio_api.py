import os
import pickle
import requests  
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import gradio as gr

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
    
    # API 주소 설정
    API_URL = "http://127.0.0.1:8000/predict"

# =========================
# 1. Load label_map (시각화용)
# =========================
label_map_path = os.path.join(Config.DATA_DIR, "label_map.pkl")
if not os.path.exists(label_map_path):
    print("[WARN] label_map.pkl not found. Charts might look weird.")
    id2emotion, id2situation = {}, {}
    emotion2id, situation2id = {}, {}
    emo_names, sit_names = [], []
else:
    with open(label_map_path, "rb") as f:
        label_map = pickle.load(f)
    
    id2emotion   = label_map["id2emotion"]
    id2situation = label_map["id2situation"]
    emotion2id   = label_map["emotion2id"]
    situation2id = label_map["situation2id"]
    
    emo_names = [id2emotion[i] for i in range(len(emotion2id))]
    sit_names = [id2situation[i] for i in range(len(situation2id))]

# =========================
# 2. API Request Logic (핵심 변경 부분)
# =========================
def request_prediction(text: str):
    """FastAPI 서버에 POST 요청을 보냅니다."""
    try:
        response = requests.post(Config.API_URL, json={"text": text})
        
        if response.status_code == 200:
            return response.json()
        else:
            print("API Error:", response.text)
            return None
    except Exception as e:
        print("Connection Error:", e)
        return None

# =========================
# 3. History & Plot
# =========================
def append_turn(history, user_text, api_result):
    if not api_result:
        return history

    turn = len(history) + 1
    
    # API 결과에서 텍스트 라벨을 가져옴
    emo_label = api_result["emotion"]
    sit_label = api_result["situation"]
    
    # 그래프를 그리기 위해 ID로 변환 (label_map 활용)
    emo_id = emotion2id.get(emo_label, -1)
    sit_id = situation2id.get(sit_label, -1)

    history.append({
        "turn": turn,
        "ts": datetime.now().strftime("%H:%M:%S"), 
        "text": user_text,
        "emotion": emo_label,
        "emo_conf": round(api_result["emotion_conf"], 2),
        "situation": sit_label,
        "sit_conf": round(api_result["situation_conf"], 2),
        "emo_id": emo_id, # 그래프용
        "sit_id": sit_id  # 그래프용
    })
    return history

def draw_timeline(history):
    if not history:
        return None, None
    
    df = pd.DataFrame(history)
    
    # Emotion Timeline
    fig_emo = plt.figure(figsize=(12, 4))
    if not df.empty and "emo_id" in df.columns:
        sizes = df["emo_conf"] * 300 
        plt.scatter(df["turn"], df["emo_id"], s=sizes, c=df["emo_id"], cmap="tab10", alpha=0.7, vmin=0, vmax=len(emo_names)-1)
        plt.yticks(range(len(emo_names)), emo_names)
    plt.xlabel("Turn")
    plt.title("Emotion Flow")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Situation Timeline
    fig_sit = plt.figure(figsize=(12, 4))
    if not df.empty and "sit_id" in df.columns:
        sizes_sit = df["sit_conf"] * 300
        plt.scatter(df["turn"], df["sit_id"], s=sizes_sit, c=df["sit_id"], cmap="Set2", alpha=0.7, vmin=0, vmax=len(sit_names)-1)
        plt.yticks(range(len(sit_names)), sit_names)
    plt.xlabel("Turn")
    plt.title("Situation Flow")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    return fig_emo, fig_sit

# =========================
# 4. UI Event Handlers
# =========================
def on_submit(text, history):
    if not text.strip():
        return "", history, None, None, None, pd.DataFrame()

    # [CHANGE] 로컬 모델 추론 -> API 요청
    api_result = request_prediction(text)
    
    if api_result is None:
        # 에러 발생 시 처리 (사용자에게 알림 등)
        return text, history, "Error", "Error", None, None, pd.DataFrame(history)

    # 히스토리 업데이트
    new_history = append_turn(history, text, api_result)
    
    # 차트 그리기
    fig_emo, fig_sit = draw_timeline(new_history)
    
    # 테이블용 DF (ID 컬럼 제외)
    df_show = pd.DataFrame(new_history).drop(columns=["emo_id", "sit_id"], errors="ignore")
    
    return "", new_history, api_result["emotion_probs"], api_result["situation_probs"], fig_emo, fig_sit, df_show

def on_reset():
    return [], None, None, None, None, pd.DataFrame()

# =========================
# 5. Gradio Build
# =========================
def build_app():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # 🎋 Mind Log: Client Mode
            ### "API와 연결된 AI 대나무숲"
            """
        )
        
        state_history = gr.State([])

        with gr.Tabs():
            with gr.TabItem("📝 기록하기 (Record)"):
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(label="Message", placeholder="Enter text...", lines=5)
                        btn_submit = gr.Button("기록하기", variant="primary")
                
                gr.Markdown("---") 
                gr.Markdown("### 🔍 분석 결과 (from API)")

                with gr.Row():
                    with gr.Column(scale=1):
                        out_emo_label = gr.Label(label="감정 (Emotion)", num_top_classes=3)
                    with gr.Column(scale=1):
                        out_sit_label = gr.Label(label="상황 (Situation)", num_top_classes=3)

            with gr.TabItem("📊 대시보드 (Dashboard)"):
                plot_emo = gr.Plot(label="Emotion Timeline")
                plot_sit = gr.Plot(label="Situation Timeline")
                
            with gr.TabItem("📂 데이터 (History)"):
                btn_reset = gr.Button("🗑️ 초기화", size="sm", variant="stop")
                history_table = gr.Dataframe(interactive=False)

        # Event Linking
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
        btn_reset.click(
            fn=on_reset,
            inputs=[],
            outputs=[state_history, out_emo_label, out_sit_label, plot_emo, plot_sit, history_table]
        )

    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860)