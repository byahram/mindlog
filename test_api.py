import requests
import json

# 1. API 주소
url = "http://127.0.0.1:8000/predict"

# 2. 보낼 데이터 (입력 텍스트)
data = {
    "text": "면접을 봤는데 너무 긴장해서 말을 잘 못한 것 같아.. 걱정돼."
}

# 3. POST 요청 보내기
try:
    response = requests.post(url, json=data)
    
    # 4. 결과 확인
    if response.status_code == 200:
        result = response.json()
        print("[SUCCEED] 성공!")
        print("---------------------------")
        print(f"입력: {data['text']}")
        print(f"감정: {result['emotion']} ({result['emotion_conf']:.2f})")
        print(f"상황: {result['situation']} ({result['situation_conf']:.2f})")
        print("---------------------------")
        # 전체 데이터 확인용
        # print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("[ERROR] 실패:", response.status_code, response.text)

except Exception as e:
    print("서버 연결 실패 (서버가 켜져 있나요?):", e)