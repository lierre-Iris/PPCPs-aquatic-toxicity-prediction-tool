"""Training and CAS lookup for the supplied PPCP random-forest dataset."""

from pathlib import Path
import re

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FEATURES = [f"x{i}" for i in range(1, 10)]
DESCRIPTORS = ["AAC", "SpPosA_B(m)", "SpMax5_Bh(m)", "P_VSA_v_3"]
ENDPOINTS = [
    "BMC05", "BMC20", "ECO", "EC05", "EC10", "EC100", "EC16", "EC20",
    "EC25", "EC50", "EC80", "EC84", "EC90", "ER50", "ET50", "IC10",
    "IC20", "IC25", "IC50", "LC50", "LD50", "LOEC", "NOEC", "NR-LETH",
]
SPECIES = {1: "Algae", 2: "Amphibians", 3: "Crustaceans", 4: "Fish", 5: "Insects"}
EXPOSURE = {1: "Aquatic", 2: "Flow-through", 3: "Lotic",
            4: "Pulse", 5: "Renewal", 6: "Static"}


def valid_cas(value: str) -> bool:
    """CAS format and check digit; membership is checked separately."""
    if not re.fullmatch(r"\d{2,7}-\d{2}-\d", value):
        return False
    digits = "".join(value.split("-")[:-1])
    return sum((i + 1) * int(ch) for i, ch in enumerate(reversed(digits))) % 10 == int(value[-1])


def load_data():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    lookup = pd.read_csv(DATA / "cas_descriptors.csv", dtype={"CAS": str})
    if train[FEATURES + ["y"]].isna().any().any() or lookup[["CAS"] + DESCRIPTORS].isna().any().any():
        raise ValueError("训练数据或 CAS 对照表有缺失值。")
    if lookup.CAS.duplicated().any():
        raise ValueError("CAS 对照表存在重复 CAS，无法唯一查找描述符。")
    return train, test, lookup.set_index("CAS")


def train_model(train: pd.DataFrame) -> RandomForestRegressor:
    """A reproducible new RF fit; original RF settings cannot be recovered from xlsx."""
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=1, max_features=1.0,
        random_state=42, n_jobs=-1,
    )
    model.fit(train[FEATURES], train["y"])
    return model


def predict(model, lookup, cas: str, species: int, exposure: int,
            medium: int, endpoint: str, duration: float):
    cas = cas.strip()
    if not valid_cas(cas):
        raise ValueError("CAS 格式或校验位不正确，请核对原始 CAS 号。")
    if cas not in lookup.index:
        raise ValueError("该 CAS 不在随附描述符表中；请先补充四种描述符后重新部署。")
    if species not in SPECIES or exposure not in EXPOSURE or medium not in (1, 2, 3):
        raise ValueError("物种组、暴露类型或介质类型编码不在建模数据范围内。")
    if endpoint not in ENDPOINTS:
        raise ValueError("终点类型不在建模数据范围内。")
    if not 0 <= duration <= 4:
        raise ValueError("观测时长须在训练数据范围 0–4 之内。")
    values = lookup.loc[cas, DESCRIPTORS]
    row = pd.DataFrame([[species, exposure, medium, ENDPOINTS.index(endpoint) + 1,
                         duration, *[float(values[name]) for name in DESCRIPTORS]]], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    return prediction, dict(zip(DESCRIPTORS, (float(values[k]) for k in DESCRIPTORS)))
