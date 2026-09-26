"""PPCPs aquatic effect concentration predictor."""

import base64
import csv
from io import StringIO
from pathlib import Path

import streamlit as st

from model import ENDPOINTS, EXPOSURE, SPECIES, load_data, predict, train_model


ROOT = Path(__file__).resolve().parent
MAX_DURATION_DAYS = {1: 3.0, 2: 4.0, 3: 2.0, 4: 4.0, 5: 2.0}
MEDIA = {1: "Fresh water", 2: "Not reported", 3: "Salt water"}


@st.cache_resource(show_spinner="首次打开时正在训练随机森林模型…")
def get_model():
    train, _, lookup = load_data()
    return train_model(train), lookup


def show_header():
    image_path = ROOT / "asset" / "asset.png"
    try:
        image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    except OSError:
        st.title("PPCPs 水生效应浓度预测")
        st.caption("输入 CAS 和五项试验条件；系统自动查找四种分子描述符。")
        return

    st.markdown(
        f"""
        <style>
        .ppcp-hero {{
            background-image: linear-gradient(90deg,
                rgba(247, 250, 252, .94) 0%,
                rgba(247, 250, 252, .82) 55%,
                rgba(247, 250, 252, .25) 100%),
                url("data:image/png;base64,{image_base64}");
            background-size: cover;
            background-position: center;
            box-sizing: border-box;
            width: 100vw;
            margin-left: calc(50% - 50vw);
            min-height: 230px;
            padding: 38px max(24px, calc((100vw - 900px) / 2));
            margin-bottom: 32px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .ppcp-hero h1 {{
            color: #183247;
            font-size: clamp(2rem, 3vw, 2.8rem);
            line-height: 1.2;
            margin: 0 0 14px 0;
        }}
        .ppcp-hero p {{
            color: #38576a;
            font-size: 1.08rem;
            line-height: 1.5;
            margin: 0;
            max-width: 850px;
        }}
        </style>
        <div class="ppcp-hero">
            <h1>PPCPs 水生效应浓度预测</h1>
            <p>输入 CAS 和五项试验条件；系统从随附的 CAS 表自动查找四种分子描述符。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_csv(cas, species, exposure, medium, endpoint, duration,
               log_value, concentration, descriptors):
    columns = [
        "CAS", "species_group", "exposure_type", "media_type",
        "endpoint_type", "duration_days", "predicted_log10_mg_L",
        "predicted_mg_L", *descriptors,
    ]
    values = [
        cas.strip(), species, exposure, medium, endpoint, duration,
        log_value, concentration, *descriptors.values(),
    ]
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(columns)
    writer.writerow(values)
    return output.getvalue().encode("utf-8-sig")


st.set_page_config(page_title="PPCPs 水生效应浓度预测", page_icon="🧪", layout="centered")
show_header()
