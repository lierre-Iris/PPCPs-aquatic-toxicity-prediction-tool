"""Streamlit entrypoint: streamlit run app.py"""

import base64
from pathlib import Path

import streamlit as st

from model import ENDPOINTS, EXPOSURE, SPECIES, load_data, predict, train_model


st.set_page_config(page_title="PPCPs 水生效应浓度预测", page_icon="🧪", layout="centered")
background_path = Path(__file__).parent / "assets" / "asset.png"
background_base64 = base64.b64encode(background_path.read_bytes()).decode()

st.markdown(
    f"""
    <style>
    .ppcp-hero {{
        background-image:
            linear-gradient(
                90deg,
                rgba(247, 250, 252, 0.96) 0%,
                rgba(247, 250, 252, 0.88) 62%,
                rgba(247, 250, 252, 0.35) 100%
            ),
            url("data:image/jpeg;base64,{background_base64}");
        background-size: cover;
        background-position: center;
        border-radius: 20px;
        padding: 36px 32px;
        margin-bottom: 28px;
        min-height: 210px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}

    .ppcp-hero h1 {{
        color: #183247;
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.18;
        margin: 0 0 16px 0;
    }}

    .ppcp-hero p {{
        color: #38576a;
        font-size: 1.08rem;
        line-height: 1.6;
        margin: 0;
    }}
    </style>

    <div class="ppcp-hero">
        <h1>PPCPs 水生效应浓度预测</h1>
        <p>输入 CAS 和五项试验条件；系统从随附的 CAS 表自动查找四种分子描述符。</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="首次打开时正在训练随机森林模型…")
def get_model():
    train, _, lookup = load_data()
    return train_model(train), lookup


try:
    model, lookup = get_model()
except (OSError, ValueError) as exc:
    st.error(f"无法加载数据或模型：{exc}")
    st.stop()

species = st.selectbox(
    "物种组（Species Group）",
    list(SPECIES),
    format_func=lambda n: f"{n} · {SPECIES[n]}",
)

# 1 藻类；2 两栖动物；3 甲壳类；4 鱼类；5 昆虫
MAX_DURATION_DAYS = {1: 3.0, 2: 4.0, 3: 2.0, 4: 4.0, 5: 2.0}
max_duration = MAX_DURATION_DAYS[species]

MEDIA = {
    1: "Fresh water",
    2: "Not reported",
    3: "Salt water",
}

with st.form("prediction"):
    cas = st.text_input("CAS 号", placeholder="例如 50-06-6")

    exposure = st.selectbox(
        "暴露类型（Exposure Type）",
        list(EXPOSURE),
        format_func=lambda n: f"{n} · {EXPOSURE[n]}",
    )

    medium = st.selectbox(
        "介质类型（Media Type）",
        [1, 2, 3],
        format_func=lambda n: f"{n} · {MEDIA[n]}",
    )

    endpoint = st.selectbox(
        "终点类型（Endpoint Type）",
        ENDPOINTS,
        index=19,
        format_func=lambda item: f"{item}（编码 {ENDPOINTS.index(item) + 1}）",
    )

    duration = st.number_input(
        "平均观测时长（Observed Duration Mean，days）",
        min_value=0.0,
        max_value=max_duration,
        value=1.0,
        step=0.01,
    )
    st.caption(f"当前物种组的平均观测时长上限：{max_duration:g} days。")

    submitted = st.form_submit_button("预测", type="primary")

if submitted:
    try:
        log_value, descriptors = predict(
            model, lookup, cas, species, exposure,
            medium, endpoint, duration
        )
    except ValueError as exc:
        st.error(str(exc))
    else:
        concentration_mg_l = 10.0 ** log_value

        with st.container(border=True):
            st.subheader("预测结果")
            st.metric(
                "预测效应浓度",
                f"{concentration_mg_l:.4g} mg/L",
            )
            st.write(f"对应的 log10 值：**{log_value:.4f}**")
            st.caption(
                f"结果对应的终点为 {endpoint}；"
                f"物种组为 {SPECIES[species]}。"
            )

        with st.expander("查看 CAS 自动匹配的分子描述符"):
            st.dataframe(
                {
                    "描述符": list(descriptors),
                    "数值": list(descriptors.values()),
                },
                hide_index=True,
                use_container_width=True,
            )

st.divider()
st.caption(
    "本程序根据给定训练数据重新拟合随机森林；原 RF 的超参数未随附件提供，"
    "因此预测结果不等同于原表 y1。模型输出为效应浓度（mg/L）的 log10 值，"
    "页面同时显示按 10 的幂反变换后的浓度（mg/L）。"
    "不同终点表示不同效应水平，结果应连同终点及试验条件一起解释。"
)
