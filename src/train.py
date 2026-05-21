"""각 task × sensor 조합으로 모델 학습 (세션 단위 split)."""
import re
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from .config import MODELS_DIR, RANDOM_STATE, TASKS, TEST_SIZE, XGB_PARAMS
from .dataset_builder import build_task_dataset


def session_id_from_filename(fname: str) -> str:
    """
    파일명에서 측정 세션 식별자(분 단위) 추출.
    같은 세션의 인접 파일들이 train/test에 섞이는 걸 방지.
    
    예: STFMK-20201105-LW15-2055_20201125_141245_004.csv
        → 'LW15-2055_20201125_1412'  (초/순번 제거)
    """
    # 진동 패턴
    m = re.search(r"-(LW\d+-\d+)_(\d{8})_(\d{4})\d{2}_", fname)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    # 전류 패턴
    m = re.search(r"-(\d+-\d+)_(\d{8})_(\d{4})\d{2}_", fname)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return fname  # 패턴 매칭 실패 시 fallback


def session_split(df: pd.DataFrame, test_size: float, random_state: int):
    """세션 단위로 train/test 분할. 같은 세션은 한쪽에만."""
    df = df.copy()
    df["session"] = df["file"].apply(session_id_from_filename)
    
    train_idx, test_idx = [], []
    rng = np.random.RandomState(random_state)
    
    for label in df["label"].unique():
        sub = df[df["label"] == label]
        sessions = sub["session"].unique().tolist()
        rng.shuffle(sessions)
        n_test = max(1, int(len(sessions) * test_size))
        test_sessions = set(sessions[:n_test])
        
        for idx, row in sub.iterrows():
            if row["session"] in test_sessions:
                test_idx.append(idx)
            else:
                train_idx.append(idx)
    
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def train_one(task_name: str, sensor: str):
    print(f"\n{'=' * 60}")
    print(f" [{sensor.upper():^10}] {task_name}")
    print('=' * 60)

    df = build_task_dataset(task_name, sensor)

    if df.empty:
        print("  ❌ 빈 데이터셋 → 건너뜀")
        return None
    if df["label"].nunique() < 2:
        print("  ❌ 라벨이 하나뿐 → 건너뜀")
        return None

    n_normal = int((df.label == 0).sum())
    n_fault = int((df.label == 1).sum())
    print(f"  정상 {n_normal}개 | 고장 {n_fault}개 | 특징 {df.shape[1] - 2}개")

    # 세션 단위 split
    train_df, test_df = session_split(df, TEST_SIZE, RANDOM_STATE)
    
    n_train_sessions = train_df["session"].nunique()
    n_test_sessions = test_df["session"].nunique()
    print(f"  세션 분할: train {len(train_df)}샘플/{n_train_sessions}세션 | "
          f"test {len(test_df)}샘플/{n_test_sessions}세션")
    
    drop_cols = ["label", "file", "session"]
    X_train = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)
    y_train = train_df["label"]
    y_test = test_df["label"]

    if y_test.nunique() < 2:
        print("  ⚠ test에 라벨이 하나뿐 → 평가 불가. 건너뜀")
        return None

    # 클래스 불균형 보정
    n_train_normal = int((y_train == 0).sum())
    n_train_fault = int((y_train == 1).sum())
    scale_pos_weight = n_train_normal / max(n_train_fault, 1)
    
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n  [Classification Report]")
    print(classification_report(
        y_test, y_pred,
        target_names=["정상", task_name],
        digits=3,
    ))

    cm = confusion_matrix(y_test, y_pred)
    print("  [Confusion Matrix]  행=실제, 열=예측")
    print(f"           예측정상  예측고장")
    print(f"  실제정상   {cm[0][0]:>6}   {cm[0][1]:>6}")
    print(f"  실제고장   {cm[1][0]:>6}   {cm[1][1]:>6}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{sensor}__{task_name}.pkl"
    joblib.dump({
        "model": model,
        "features": list(X_train.columns),
        "task": task_name,
        "sensor": sensor,
    }, model_path)
    print(f"\n  💾 저장: {model_path.name}")
    return model


def main():
    print("\n" + "=" * 60)
    print(" 회전기기 고장 진단 모델 학습 (세션 단위 split)")
    print("=" * 60)

    for task in TASKS:
        for sensor in ["vibration", "current"]:
            train_one(task, sensor)

    print("\n" + "=" * 60)
    print(" ✅ 학습 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()