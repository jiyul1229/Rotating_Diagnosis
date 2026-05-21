"""파일 시스템을 스캔해서 task별 DataFrame을 만듦."""
import pandas as pd

from .config import VIB_ROOT, CUR_ROOT, TASKS
from .feature_extraction import (
    extract_vibration_features,
    extract_current_features,
)


def build_task_dataset(task_name: str, sensor: str) -> pd.DataFrame:
    """
    한 task의 한 센서 데이터셋.
    
    Args:
        task_name: TASKS의 키 (예: "축정렬불량")
        sensor: "vibration" 또는 "current"
    
    Returns:
        feature 컬럼들 + label(0=정상, 1=고장) + file
    """
    if task_name not in TASKS:
        raise KeyError(task_name)

    info = TASKS[task_name]
    motor, capacity = info["motor"], info["capacity"]

    if sensor == "vibration":
        root, extract = VIB_ROOT, extract_vibration_features
    elif sensor == "current":
        root, extract = CUR_ROOT, extract_current_features
    else:
        raise ValueError(f"sensor must be 'vibration' or 'current': {sensor}")

    records = []
    fail_log = []

    for label_name, label_id in [("정상", 0), (task_name, 1)]:
        folder = root / capacity / motor / label_name
        if not folder.exists():
            print(f"  ⚠ 폴더 없음: {folder}")
            continue

        files = sorted(list(folder.glob("*.csv")) + list(folder.glob("*.CSV")))
        print(f"  [{sensor}/{motor}/{label_name}] 파일 {len(files)}개")

        for f in files:
            try:
                feats = extract(f)
                feats["label"] = label_id
                feats["file"] = f.name
                records.append(feats)
            except Exception as e:
                fail_log.append((f.name, str(e)))

    if fail_log:
        print(f"  ❌ 실패 {len(fail_log)}개. 처음 3개:")
        for name, err in fail_log[:3]:
            print(f"      {name}: {err}")

    return pd.DataFrame(records)
