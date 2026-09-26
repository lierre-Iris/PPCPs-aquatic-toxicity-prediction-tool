"""PPCP aquatic effect-concentration predictor: Streamlit entrypoint."""

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from model import ENDPOINTS, EXPOSURE, SPECIES, load_data, predict, train_model, valid_cas


ROOT = Path(__file__).resolve().parent
MAX_DURATION_DAYS = {1: 3.0, 2: 4.0, 3: 2.0, 4: 4.0, 5: 2.0}
MEDIA = {1: "Fresh water", 2: "Not reported", 3: "Salt water"}
EXPOSURE_LABELS = {**EXPOSURE, 1: "Aquatic"}
BATCH_COLUMNS = ["CAS", "species_group", "exposure_type", "media_type",
                 "endpoint_type", "duration_days"]


@st.cache_resource(show_spinner="首次打开时正在训练随机森林模型…")
def get_model():
    train, _, lookup = load_data()
    return train_model(train), lookup


def show_header():
    candidates = [
        ROOT / "asset" / "asset.png",
        ROOT / "assets" / "asset.png",
        ROOT / "assets" / "ppcp_background.jpg",
    ]
    image = next((path for path in candidates if path.is_file()), None)
    title = "PPCPs 水生效应浓度预测"
    intro = "输入 CAS 和试验条件，自动查找四种分子描述符并预测相应终点的效应浓度。"
    if image is None:
        st.title(title)
        st.caption(intro)
        return
    mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <style>
        .ppcp-hero {{
            background-image: linear-gradient(90deg,
                rgba(247,250,252,.96) 0%, rgba(247,250,252,.88) 57%,
                rgba(247,250,252,.30) 100%),
                url("data:{mime};base64,{encoded}");
            background-size: cover;
            background-position: center;
            box-sizing: border-box;
            width: 100vw;
            margin-left: calc(50% - 50vw);
            min-height: 230px;
            padding: 38px max(24px, calc((100vw - 900px) / 2));
            margin-bottom: 28px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .ppcp-hero h1 {{ color: #183247; font-size: clamp(2rem, 3vw, 2.8rem);
                         line-height: 1.2; margin: 0 0 14px; }}
        .ppcp-hero p {{ color: #38576a; font-size: 1.08rem;
                        line-height: 1.5; margin: 0; max-width: 850px; }}
        </style>
        <div class="ppcp-hero"><h1>{title}</h1><p>{intro}</p></div>
        """,
        unsafe_allow_html=True,
    )


def show_result(log_value, descriptors, cas, species, endpoint, exposure, medium, duration):
    concentration = 10.0 ** log_value
    with st.container(border=True):
        st.subheader("预测结果")
        st.metric("预测效应浓度", f"{concentration:.4g} mg/L")
        st.write(f"对应的 log10 值：**{log_value:.4f}**")
        st.caption(
            f"CAS {cas} · {endpoint} · {SPECIES[species]} · "
            f"{EXPOSURE_LABELS[exposure]} · {MEDIA[medium]} · {duration:g} days"
        )
    with st.expander("查看自动匹配的四种分子描述符"):
        st.dataframe(
            pd.DataFrame({"描述符": list(descriptors), "数值": list(descriptors.values())}),
            hide_index=True, use_container_width=True,
        )
    st.caption("浓度是模型对所选终点及试验条件的预测，不同终点的效应定义不同。")


def batch_predict(model, lookup, source):
    missing = [name for name in BATCH_COLUMNS if name not in source.columns]
    if missing:
        raise ValueError("CSV 缺少列：" + "、".join(missing))
    if source.empty or len(source) > 200:
        raise ValueError("每次请上传 1–200 行数据。")
    rows = []
