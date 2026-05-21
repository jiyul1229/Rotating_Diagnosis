# 회전기기 고장 진단 (Rotating Machinery Fault Diagnosis)

AI Hub 회전기기 진동/전류 데이터를 기반으로 4가지 고장 유형을 이진 분류(정상 vs 고장)하는 XGBoost 모델 학습 파이프라인.

## 프로젝트 구조

```
rotating_diagnosis/
├── src/
│   ├── __init__.py
│   ├── config.py              # 전역 설정 (경로, 학습 파라미터, task 정의)
│   ├── feature_extraction.py  # 진동/전류 신호 → 통계 특징 추출
│   ├── dataset_builder.py     # 파일 시스템 스캔 → task별 DataFrame 생성
│   └── train.py               # 모델 학습 및 평가 (최종 실행 파일)
├── data/aihub/                # AI Hub 원본 데이터
│   ├── vibration/             # 진동 센서 CSV
│   └── current/               # 전류 센서 CSV
└── models/                    # 학습된 모델 저장 (.pkl)
```

## 각 모듈 설명

### `config.py` — 전역 설정

프로젝트 전체에서 사용하는 경로와 하이퍼파라미터를 정의한다.

- **데이터 경로**: `~/rotating_diagnosis/data/aihub/` 하위의 `vibration/`, `current/` 디렉토리
- **4개 진단 task** (모두 2.2kW 모터):

| Task | 모터 ID | 설명 |
|------|---------|------|
| 축정렬불량 | L-DSF-01 | 축 정렬 불량 진단 |
| 회전체불평형 | L-EF-04 | 회전체 불평형 진단 |
| 베어링불량 | L-SF-04 | 베어링 불량 진단 |
| 벨트느슨함 | R-EF-05 | 벨트 느슨함 진단 |

- **XGBoost 파라미터**: `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`
- **데이터 분할**: test 20%, random_state=42

### `feature_extraction.py` — 특징 추출

CSV 원본 신호에서 시간 도메인 통계 특징을 추출한다.

- **공통 처리** (`_read_signal`): CSV의 상위 9줄 메타데이터 스킵, trailing comma로 생긴 NaN 컬럼 제거, 숫자 변환
- **진동 특징** (`extract_vibration_features`): 1축 신호 → **8개 통계** (mean, std, rms, max, min, ptp, kurtosis, skewness)
- **전류 특징** (`extract_current_features`): 3상(U/V/W) 각각 8개 통계 → **총 24개 특징**

### `dataset_builder.py` — 데이터셋 구성

파일 시스템의 폴더 구조(`용량/모터ID/라벨명/`)를 스캔하여 특징 추출 후 DataFrame으로 합친다.

- task별로 "정상"(label=0) / "고장명"(label=1) 폴더의 CSV 파일들을 순회
- 각 파일에 대해 `feature_extraction`의 함수를 호출하여 특징 벡터 생성
- 실패 파일은 로그로 출력하고 건너뜀

### `train.py` — 모델 학습 (최종 실행)

4개 task × 2개 센서 = **총 8개 모델**을 학습한다.

핵심 로직:
1. **세션 단위 split**: 파일명에서 측정 세션 ID(분 단위)를 추출하여, 같은 세션의 데이터가 train/test에 동시에 들어가지 않도록 분할 (데이터 누수 방지)
2. **클래스 불균형 보정**: `scale_pos_weight`로 정상/고장 비율 자동 보정
3. **평가**: Classification Report + Confusion Matrix 출력
4. **모델 저장**: `models/` 디렉토리에 `{sensor}__{task_name}.pkl` 형태로 저장 (모델 + feature 목록 + 메타정보)

## 실행 방법

```bash
# 프로젝트 루트에서
python -m src.train
```

## 필요 패키지

- numpy
- pandas
- scikit-learn
- xgboost
- joblib
