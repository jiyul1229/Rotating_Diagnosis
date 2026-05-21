"""진동/전류 신호에서 시간 도메인 통계 특징을 추출."""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .config import META_LINES


def _read_signal(file_path: Path) -> pd.DataFrame:
    file_path = unicodedata.normalize("NFC", str(file_path))
    df = pd.read_csv(file_path, skiprows=META_LINES, header=None)
    # trailing comma로 생긴 전체 NaN 컬럼 제거
    df = df.dropna(axis=1, how="all")
    # 숫자 변환
    df = df.apply(pd.to_numeric, errors="coerce")
    # 모든 셀이 NaN인 행만 제거 (특정 셀 NaN으로 행 전체를 날리지 않음)
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def _stats(values: np.ndarray, prefix: str) -> dict:
    """한 채널의 시간 도메인 통계."""
    s = pd.Series(values)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std":  float(np.std(values)),
        f"{prefix}_rms":  float(np.sqrt(np.mean(values ** 2))),
        f"{prefix}_max":  float(np.max(values)),
        f"{prefix}_min":  float(np.min(values)),
        f"{prefix}_ptp":  float(np.ptp(values)),
        f"{prefix}_kurt": float(s.kurt()),
        f"{prefix}_skew": float(s.skew()),
    }


def extract_vibration_features(file_path: Path) -> dict:
    """진동 1축 → 통계 8개."""
    df = _read_signal(file_path)
    if len(df) < 10 or df.shape[1] < 2:
        raise ValueError(f"진동 데이터 형식 오류: shape={df.shape}")
    values = df.iloc[:, 1].values
    return _stats(values, "vib")


def extract_current_features(file_path: Path) -> dict:
    """전류 3상(U/V/W) → 통계 24개."""
    df = _read_signal(file_path)
    if len(df) < 10 or df.shape[1] < 4:
        raise ValueError(f"전류 데이터 형식 오류: shape={df.shape}")
    
    features = {}
    for i, phase in enumerate(["u", "v", "w"]):
        values = df.iloc[:, i + 1].values  # 0번째는 시간
        features.update(_stats(values, f"cur_{phase}"))
    return features
