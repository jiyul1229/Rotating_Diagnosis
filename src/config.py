"""프로젝트 전역 설정."""
from pathlib import Path

# 경로
PROJECT_ROOT = Path.home() / "rotating_diagnosis"
DATA_ROOT = PROJECT_ROOT / "data" / "aihub"
VIB_ROOT = DATA_ROOT / "vibration"
CUR_ROOT = DATA_ROOT / "current"
MODELS_DIR = PROJECT_ROOT / "models"

# 4개 task (2.2kW 우선, 55kW는 옵션)
TASKS = {
    "축정렬불량":  {"motor": "L-DSF-01",   "capacity": "2.2kW"},
    "회전체불평형": {"motor": "L-EF-04",    "capacity": "2.2kW"},
    "베어링불량":  {"motor": "L-SF-04",    "capacity": "2.2kW"},
    "벨트느슨함":  {"motor": "R-EF-05",    "capacity": "2.2kW"},
}

# 메타데이터 행 수
META_LINES = 9

# 학습 파라미터
TEST_SIZE = 0.2
RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "logloss",
}
