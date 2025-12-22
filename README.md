# 🎋 Mind Log: Multi-Task Emotion Tracker

**"당신의 감정을 비워드리는 AI 가비지 컬렉터 (Emotional Garbage Collector)"**

_말하지 않아도 드러나는 감정의 흐름을 데이터로 기록하다_

<br>

## 1. Background & Introduction

**Mind Log**는 사용자의 구어체 텍스트를 분석하여 **'감정(Emotion)'** 과 **'상황(Situation)'** 을 동시에 추론하는 **Multi-Task Learning (MTL) 기반의 AI 다이어리** 입니다. 무거운 LLM 대신 경량화된 PLM(Pre-trained Language Model)을 사용하여 빠른 추론 속도와 효율성을 확보했습니다.

현대인들은 부정적인 감정을 털어놓고 싶어 하지만, 지인에게 말하기엔 부담스럽고 SNS에는 기록이 남을까 우려합니다. 기존의 챗봇들은 기계적인 위로("힘내세요")를 반복하여 오히려 피로감을 주기도 합니다.

Mind Log는 **"침묵의 기록(Silent Tracker)"** 을 지향합니다. 사용자가 뱉어낸 감정적 언어를 AI가 객관적으로 분석하고 분류해 줌으로써, **"내 말이 이해받았다"는 심리적 위로를 제공하고 감정 해소를 돕는 'AI 대나무숲' 서비스** 입니다.

<br>

## 2. Key Features

- **Multi-Task Learning**
  - 단일 모델로 감정과 상황을 동시에 분류하여 메모리 효율성 극대화.
- **Real-time Inference**
  - LLM API 없이 로컬 환경에서 즉각적인 분석 가능 (Latency < 50ms).
- **Interactive UI**
  - Gradio 기반의 대시보드 제공 (입력 탭 / 감정 흐름 타임라인 시각화 탭).
- **Robust to Colloquialism**
  - 문어체가 아닌 뉴스 댓글 기반 모델(`KcELECTRA`)을 사용하여 구어체(반말, 은어 등) 처리 능력 강화.

<br>

## 3. Tech Stack & Architecture

### 3-1. Multi-Task Learning (MTL) Architecture

단일 입력 문장에 대해 두 가지 라벨(감정, 상황)을 동시에 예측하기 위해 **Hard Parameter Sharing** 방식을 적용했습니다.

- **Shared Encoder**: `KcELECTRA-Base` (Body)를 공유하여 문맥 정보를 학습합니다.
- **Independent Heads**: 감정 분류기(Emotion Head)와 상황 분류기(Situation Head)를 분리하여, 서로 다른 태스크가 공통된 특징을 학습하면서도 개별적인 판단을 내리도록 설계했습니다.
- **Optimization**: `Weighted CrossEntropy Loss`를 적용하여 데이터 불균형 문제(Class Imbalance)를 해결하고, 소수 클래스(예: 당황, 상처)의 예측 성능을 높였습니다.

### 3-2. Model Selection Strategy

- **Challenge**: 초기 베이스라인(`KoBERT`) 사용 시 정확도가 **37%** 수준에 그침. 위키피디아 기반의 문어체 모델은 일상 대화의 뉘앙스를 파악하는 데 한계가 있었음.
- **Solution**: 뉴스 댓글 데이터(1억 건 이상)로 학습되어 구어체, 신조어, 오탈자에 강한 `beomi/KcELECTRA-base` 모델로 백본을 교체.
- **Result**: 모델 구조 변경 없이 백본 교체만으로 성능이 **2배 이상 향상**됨을 확인.

<br>

## 4. Data Engineering & EDA

- **Source**: AI Hub '감성 대화 말뭉치' (약 4만 건)
- **Preprocessing Strategy**
  - **Context Merging (문맥 병합)**: '사람문장1' 단일 발화만으로는 감정의 원인을 파악하기 어려워, `문장1 + 문장2 + 문장3`을 하나의 시퀀스로 병합하여 모델 입력으로 사용.
  - **Text Cleaning**: 정규표현식(`re`)을 사용하여 특수문자 및 불필요한 공백 제거.
  - **Class Balancing**: `compute_class_weight`를 사용하여 학습 데이터 분포의 역수를 Loss 가중치로 부여.
- **EDA Insight**
  - **Heatmap Analysis**: 특정 상황(예: 직장)과 특정 감정(예: 분노) 간의 짙은 상관관계를 확인하여 MTL 구조의 타당성 확보.

<br>

## 5. Model Performance

초기 모델(KoBERT)과 최종 모델(KcELECTRA)의 성능 비교 결과입니다. **도메인에 적합한 모델 선정(Domain Adaptation)**이 성능에 결정적인 영향을 미쳤음을 확인했습니다.

| Model                 | Task      | Accuracy         | Improvement    |
| :-------------------- | :-------- | :--------------- | :------------- |
| **KoBERT (Baseline)** | Emotion   | 0.23 (23.4%)     | -              |
| **KcELECTRA (Final)** | Emotion   | **0.77 (76.9%)** | **+53.5%p** 🔺 |
|                       |           |                  |                |
| **KoBERT (Baseline)** | Situation | 0.24 (23.8%)     | -              |
| **KcELECTRA (Final)** | Situation | **0.75 (74.6%)** | **+50.8%p** 🔺 |

1.  **Why KoBERT Failed? (Acc 23%)**: KoBERT는 위키피디아 등 정제된 **문어체** 위주로 학습되었습니다. 감성 대화 말뭉치와 같은 **구어체(반말, 은어, 감정적 표현)** 데이터에서는 맥락을 제대로 파악하지 못해 학습이 수렴하지 못했습니다.
2.  **Why KcELECTRA Succeeded? (Acc 77%)**: 뉴스 댓글 데이터로 학습된 `KcELECTRA`는 구어체와 비정형 텍스트 이해도가 매우 높습니다. 모델 교체만으로 비약적인 성능 향상을 이뤘으며, **Weighted Loss** 도입으로 클래스 불균형을 해소하여 F1-Score 또한 균형 잡힌 결과를 보였습니다.

> _Confusion Matrix를 통해 '기쁨(F1 0.97)'과 같은 명확한 감정은 완벽에 가깝게 분류함을 확인했습니다._

<br>

## 6. Getting Started

### 6-1. Installation

```bash
# Repository Clone
git clone [https://github.com/your-username/mindlog.git](https://github.com/your-username/mindlog.git)
cd mindlog

# Install Dependencies
pip install -r requirements.txt
```

### 6-2. Run Demo

```bash
# 앱 실행 (로컬 서버 127.0.0.1:7860)
python src/app.py
```

<br>

## 7. Project Structure

```bash
mindlog/
├── data/
│   ├── raw/                # 원본 데이터 (AI Hub)
│   └── processed/          # 전처리 완료 데이터 (.csv, .pkl)
├── models/                 # 학습된 모델 가중치 (.bin)
├── notebooks/              # 실험 및 EDA (Jupyter Notebook)
│   ├── 01_eda.py           # 데이터 분포 및 시각화
│   ├── 02_preprocessing.py # 전처리 파이프라인
│   └── 03_train.py         # 모델 학습 및 평가
├── src/                    # 소스 코드
│   ├── preprocess.py       # 데이터 전처리 및 인코딩
│   ├── train.py            # 모델 학습 (Trainer, Loss Function)
│   └── app.py              # Gradio 웹 애플리케이션
├── requirements.txt        # 의존성 패키지
└── README.md
```

<br />

## 8. Tech Stack

| Category      | Technology                                            |
| ------------- | ----------------------------------------------------- |
| Language      | Python 3.8+                                           |
| Model         | `beomi/KcELECTRA-base-v2022`                          |
| Framework     | PyTorch, Hugging Face Transformers                    |
| Optimization  | AdamW, Cosine Annealing Warmup, Weighted CrossEntropy |
| Visualization | Matplotlib, Seaborn                                   |
| Serving       | Gradio                                                |
