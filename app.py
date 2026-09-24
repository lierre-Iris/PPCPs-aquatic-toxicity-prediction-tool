"""Streamlit entrypoint: streamlit run app.py"""

import streamlit as st

from model import ENDPOINTS, EXPOSURE, SPECIES, load_data, predict, train_model


st.set_page_config(page_title="PPCPs 水生效应浓度预测", page_icon="🧪", layout="centered")
st.title("PPCPs 水生效应浓度预测")
st.caption("输入 CAS 和五项试验条件；系统从随附的 CAS 表自动查找四种分子描述符。")


@st.cache_resource(show_spinner="首次打开时正在训练随机森林模型…")
def get_model():
    train, _, lookup = load_data()
    return train_model(train), lookup


try:
    model, lookup = get_model()
except (OSError, ValueError) as exc:
    st.error(f"无法加载数据或模型：{exc}")
    st.stop()

with st.form("prediction"):
    cas = st.text_input("CAS 号", placeholder="例如 50-06-6")
    species = st.selectbox("物种组（Species Group）", list(SPECIES),
                           format_func=lambda n: f"{n} · {SPECIES[n]}")
    exposure = st.selectbox("暴露类型（Exposure Type）", list(EXPOSURE),
                            format_func=lambda n: f"{n} · {EXPOSURE[n]}")
    medium = st.selectbox("介质类型（Media Type）", [1, 2, 3],
                          format_func=lambda n: f"编码 {n}（原表未提供名称）")
    endpoint = st.selectbox("终点类型（Endpoint Type）", ENDPOINTS, index=19,
                            format_func=lambda item: f"{item}（编码 {ENDPOINTS.index(item) + 1}）")
    duration = st.number_input("平均观测时长（Observed Duration Mean；使用建模表原始数值单位）",
                               min_value=0.0, max_value=4.0, value=1.0, step=0.01,
                               help="附件未说明单位。直接输入与 x5 一致的数值；训练范围为 0–4。")
    submitted = st.form_submit_button("预测", type="primary")

if submitted:
    try:
        log_value, descriptors = predict(model, lookup, cas, species, exposure,
                                         medium, endpoint, duration)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.success(f"预测的 log 转换效应浓度：{log_value:.4f}")
        st.write("自动查得的描述符：")
        st.dataframe({"描述符": list(descriptors), "数值": list(descriptors.values())},
                     hide_index=True, use_container_width=True)
        st.info("原始工作簿未注明对数底数和效应浓度单位，因此此处仅报告 log 尺度预测值；"
                "核对建模时的 log 定义与原始浓度单位后，才能可靠地换算成浓度。")

st.divider()
st.caption("本程序按给定训练数据重新拟合随机森林；原 RF 的超参数未随附件提供，预测结果不等同于原表 y1。"
           "不同终点表示不同效应水平，所得浓度应连同终点及试验条件一起解释。")
