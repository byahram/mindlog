# =============================
# 1. 환경 설정 및 라이브러리 임포트
# =============================
import os
import sys
import platform
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from wordcloud import WordCloud
from konlpy.tag import Okt
from collections import Counter

# 한글 폰트 설정
system_name = platform.system()

if system_name == 'Windows':
    # 윈도우
    plt.rc("font", family="Malgun Gothic")
    font_path = "C:/Windows/Fonts/malgun.ttf" # 워드클라우드용
elif system_name == 'Darwin':
    # 맥(Mac)
    plt.rc("font", family="AppleGothic")
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc" # 워드클라우드용 (없으면 경로 수정 필요)
else:
    # 리눅스 (Colab 등)
    plt.rc("font", family="NanumBarunGothic")
    font_path = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"

plt.rcParams["axes.unicode_minus"] = False
print(f"[INFO] 운영체제 감지: {system_name}, 폰트 설정 완료.")


# ===============================================
# 2. 데이터 로드 및 통합 (Train + Validation)
# ===============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data/raw")       # 데이터 폴더
SAVE_DIR = os.path.join(BASE_DIR, "../outputs/eda_results") # 결과 저장 폴더

# 저장 폴더가 없으면 생성
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"[INFO] 결과 저장 폴더 생성: {SAVE_DIR}")

# 파일명 정의
XLSX_TRAIN_FILE = "감성대화말뭉치_Training.xlsx"
XLSX_VALID_FILE = "감성대화말뭉치_Validation.xlsx"

# 이미지 저장 헬퍼 함수
def save_fig(filename):
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f" >> 이미지 저장됨: {filename}")

# 데이터 로드 함수
def load_data(filename):
    # data 폴더 내의 파일 경로
    filepath = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"[INFO] 파일 로드 중: {filepath}")
        return pd.read_excel(filepath)
    else:
        # 데이터 폴더에 없으면 현재 폴더에서도 찾아봄
        filepath_root = os.path.join(BASE_DIR, filename)
        if os.path.exists(filepath_root):
            print(f"[INFO] 파일 로드 중 (루트 경로): {filepath_root}")
            return pd.read_excel(filepath_root)
        else:
            print(f"[ERROR] 파일을 찾을 수 없습니다: {filename}")
            print(f"      경로 확인: {DATA_DIR} 또는 {BASE_DIR}")
            return None

print("데이터 로딩 시작...")
df_train = load_data(XLSX_TRAIN_FILE)
df_valid = load_data(XLSX_VALID_FILE)

# 데이터가 하나라도 없으면 종료
if df_train is None or df_valid is None:
    print("[ERROR] 데이터 파일을 찾지 못해 종료합니다.")
    sys.exit()

# EDA를 위해 데이터 합치기
df_train["split"] = "train"
df_valid["split"] = "valid"

df_all = pd.concat([df_train, df_valid], ignore_index=True)
print(f"\n[INFO] 데이터 통합 완료. 총 데이터 개수: {len(df_all)}개")
print(f" - Train: {len(df_train)}개")
print(f" - Valid: {len(df_valid)}개")


# ==========================================
# 3. 데이터 기본 확인
# ==========================================
print("\n[데이터 구조 확인 (Head 3)]")
print(df_all.head(3))

print("\n[결측치 확인]")
print(df_all.isnull().sum())

# 필요한 컬럼만 선택
target_cols = ["연령", "성별", "상황키워드", "감정_대분류", "감정_소분류", "사람문장1"]
# 존재하는 컬럼만 가져오기 (에러 방지)
valid_cols = [c for c in target_cols if c in df_all.columns]
df_eda = df_all[valid_cols].copy()


# ==========================================
# 4. 데이터 분포 시각화 (Univariate)
# ==========================================
# (1) 감정 대분류 분포
if "감정_대분류" in df_eda.columns:
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x="감정_대분류", data=df_eda, order=df_eda["감정_대분류"].value_counts().index)
    plt.title("감정 대분류 분포 (Target Distribution)")
    plt.xlabel("감정")
    plt.ylabel("개수")

    # 막대 위에 숫자 표시
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 30, int(height), ha="center", size=9)

    save_fig("01_감정대분류_분포.png")
    # plt.show()

# (2) 연령 및 성별 분포
if "연령" in df_eda.columns and "성별" in df_eda.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(x="연령", data=df_eda, order=df_eda["연령"].value_counts().index, ax=axes[0])
    axes[0].set_title("연령대 분포")

    sns.countplot(x="성별", data=df_eda, ax=axes[1])
    axes[1].set_title("성별 분포")

    save_fig("02_연령_성별_분포.png")
    # plt.show()


# =====================
# 5. 상관관계 분석
# =====================
def plot_heatmap(col1, col2, title, filename):
    if col1 not in df_eda.columns or col2 not in df_eda.columns:
        return
        
    crosstab = pd.crosstab(df_eda[col1], df_eda[col2], normalize="index")

    plt.figure(figsize=(10, 8))
    sns.heatmap(crosstab, annot=True, fmt=".2f", cmap="Blues", linewidths=.5)
    plt.title(title)
    plt.ylabel(col1)
    plt.xlabel(col2)

    save_fig(filename)
    # plt.show()

# (1) 연령 vs 감정 대분류
plot_heatmap("연령", "감정_대분류", "연령대별 감정 분포 (비율)", "03_heatmap_연령_감정.png")

# (2) 상황키워드 vs 감정 대분류 (상위 15개 상황만)
if "상황키워드" in df_eda.columns:
    top_situations = df_eda["상황키워드"].value_counts().nlargest(15).index
    df_situation_subset = df_eda[df_eda["상황키워드"].isin(top_situations)]

    plt.figure(figsize=(12, 10))
    crosstab_sit = pd.crosstab(df_situation_subset["상황키워드"], df_situation_subset["감정_대분류"], normalize="index")
    sns.heatmap(crosstab_sit, annot=True, fmt=".2f", cmap="Reds", linewidths=.5)
    plt.title("상위 15개 상황키워드별 감정 분포")

    save_fig("04_heatmap_상황_감정.png")
    # plt.show()


# ======================
# 6. 문장 길이 분석
# ======================
# 사람문장1의 길이 분포
if "사람문장1" in df_eda.columns:
    df_eda["text_len"] = df_eda["사람문장1"].astype(str).apply(len)

    plt.figure(figsize=(12, 5))
    sns.histplot(df_eda["text_len"], bins=50, kde=True)
    plt.title("사람문장1 길이 분포 (글자 수)")
    plt.xlabel("글자 수")
    plt.axvline(x=df_eda["text_len"].mean(), color="r", linestyle="--", label=f"평균: {df_eda['text_len'].mean():.1f}")
    plt.axvline(x=np.percentile(df_eda["text_len"], 95), color="g", linestyle="--", label=f"95%: {np.percentile(df_eda['text_len'], 95):.1f}")
    plt.legend()

    save_fig("05_문장길이_분포.png")
    # plt.show()

    print(f"최대 길이: {df_eda['text_len'].max()}")
    print(f"권장 max_length (95% 커버): {int(np.percentile(df_eda['text_len'], 95))}")

# 전체 문장 (사람문장 1~3) 병합 길이 분석
# 원본 데이터프레임에서 문장 컬럼 가져오기
sentence_cols = [col for col in ["사람문장1", "사람문장2", "사람문장3"] if col in df_all.columns]
if sentence_cols:
    df_merged = df_all[sentence_cols].fillna("").astype(str)
    df_eda["merged_text"] = df_merged.agg(" ".join, axis=1).str.strip()
    df_eda["merged_len"] = df_eda["merged_text"].apply(len)

    plt.figure(figsize=(12, 5))
    sns.histplot(df_eda["merged_len"], bins=60, kde=True)

    plt.title("사람문장1~3 합친 문장 길이 분포 (글자 수)")
    plt.xlabel("글자 수")

    plt.axvline(
        x=df_eda["merged_len"].mean(),
        color="r",
        linestyle="--",
        label=f"평균: {df_eda['merged_len'].mean():.1f}"
    )

    plt.axvline(
        x=np.percentile(df_eda["merged_len"], 95),
        color="g",
        linestyle="--",
        label=f"95%: {np.percentile(df_eda['merged_len'], 95):.1f}"
    )

    plt.legend()

    save_fig("05_merged_문장길이_분포.png")
    # plt.show()

    print(f"합친 문장 최대 길이: {df_eda['merged_len'].max()}")
    print(f"합친 문장 권장 max_length (95% 범위): {int(np.percentile(df_eda['merged_len'], 95))}")


# ===========================
# 7. 워드클라우드 시각화
# ===========================
# 형태소 분석기 준비 (명사 추출용)
try:
    okt = Okt()
except Exception as e:
    print(f"[WARNING] KoNLPy 초기화 실패 (Java 미설치 등): {e}")
    okt = None

# 워드클라우드 그리는 함수 정의
def draw_wordcloud(df, target_col, target_value, title, filename):
    if okt is None:
        return

    # 해당 감정/상황에 맞는 데이터만 필터링
    filtered_df = df[df[target_col] == target_value]

    if len(filtered_df) == 0:
        print(f" >> '{target_value}' 데이터가 없습니다. 스킵합니다.")
        return

    # 텍스트 데이터 뭉치기
    text_all = " ".join(filtered_df["사람문장1"].astype(str).tolist())

    # [중요] 명사만 추출 (시간이 좀 걸립니다)
    print(f" >> '{target_value}' 데이터 명사 추출 중...")
    nouns = okt.nouns(text_all)

    # 한 글자 단어 제거
    nouns = [n for n in nouns if len(n) > 1]
    
    # 단어 빈도수 계산
    counts = Counter(nouns)

    if not counts:
        print(f" >> '{target_value}'에서 추출된 명사가 없습니다.")
        return

    # 워드클라우드 설정
    wc = WordCloud(
        font_path=font_path, # 위에서 설정한 OS별 폰트 경로
        background_color="white",
        width=800,
        height=600,
        max_words=100,
        colormap="coolwarm"
    )

    # 워드클라우드 생성
    wc.generate_from_frequencies(counts)

    # 그림 그리기 및 저장
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off") 
    plt.title(title, fontsize=15)

    # 저장
    save_fig(filename)
    # plt.show()

if "감정_대분류" in df_eda.columns and okt is not None:
    emotions = df_eda["감정_대분류"].unique()
    print(f"\n분석 대상 감정: {emotions}")

    for emo in emotions:
        print(f"\n[{emo}] 워드클라우드 생성 시작...")
        draw_wordcloud(
            df=df_eda,
            target_col="감정_대분류",
            target_value=emo,
            title=f"'{emo}' 감정의 주요 키워드",
            filename=f"wc_감정_{emo}.png"
        )

print("\n[완료] 모든 분석이 종료되었습니다. 'eda_results' 폴더를 확인하세요.")