"""PPCP aquatic effect-concentration predictor: Streamlit entrypoint."""

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from model import ENDPOINTS, EXPOSURE, SPECIES, load_data, predict, train_model, valid_cas


ROOT = Path(__file__).resolve().parent
MAX_DURATION_DAYS = {1: 3.0, 2: 4.0, 3: 2.0, 4: 4.0, 5: 2.0}
MEDIA = {1: "Fresh water", 2: "Not reported", 3: "Salt water"}
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
    try:
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    except OSError:
        st.title(title)
        st.caption(intro)
        return
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
            f"{EXPOSURE[exposure]} · {MEDIA[medium]} · {duration:g} days"
        )
    record = pd.DataFrame([{
        "CAS": cas, "species_group": species, "exposure_type": exposure,
        "media_type": medium, "endpoint_type": endpoint,
        "duration_days": duration, "predicted_log10_mg_L": log_value,
        "predicted_mg_L": concentration, **descriptors,
    }])
    st.download_button(
        "下载本次结果 CSV", record.to_csv(index=False).encode("utf-8-sig"),
        file_name="ppcp_prediction.csv", mime="text/csv",
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
    for _, item in source.iterrows():
        result = item.to_dict()
        result.update({"predicted_log10_mg_L": None, "predicted_mg_L": None, "error": ""})
        try:
            cas = str(item["CAS"]).strip()
            codes = []
            for name in ("species_group", "exposure_type", "media_type"):
                numeric = float(item[name])
                if not numeric.is_integer():
                    raise ValueError(f"{name} 必须是整数编码。")
                codes.append(int(numeric))
            species, exposure, medium = codes
            duration = float(item["duration_days"])
            if species not in MAX_DURATION_DAYS:
                raise ValueError("物种组编码必须为 1–5。")
            if not 0 <= duration <= MAX_DURATION_DAYS[species]:
                raise ValueError(f"所选物种组的时长须在 0–{MAX_DURATION_DAYS[species]:g} days。")
            endpoint = str(item["endpoint_type"]).strip()
            log_value, _ = predict(model, lookup, cas, species, exposure,
                                   medium, endpoint, duration)
            result["predicted_log10_mg_L"] = log_value
            result["predicted_mg_L"] = 10.0 ** log_value
        except (ValueError, TypeError) as exc:
            result["error"] = str(exc)
        rows.append(result)
    return pd.DataFrame(rows)


st.set_page_config(page_title="PPCPs 水生效应浓度预测", page_icon="🧪", layout="centered")
show_header()
st.info("① 选择单条或批量预测　→　② 填写 CAS 与试验条件　→　③ 查看并下载结果")
try:
    model, lookup = get_model()
except (OSError, ValueError) as exc:
    st.error(f"无法加载模型或数据：{exc}")
    st.stop()

single_tab, batch_tab, about_tab = st.tabs(["🧪 单条预测", "📄 批量预测", "📖 模型与数据"])

with single_tab:
    st.subheader("填写预测条件")
    st.caption(f"当前描述符表可匹配 {len(lookup)} 个 CAS；仅对表内化合物提供预测。")
    cas = st.text_input("CAS 号", placeholder="例如 50-06-6", key="single_cas")
    if cas.strip():
        if not valid_cas(cas.strip()):
            st.warning("CAS 格式或校验位不正确。")
        elif cas.strip() not in lookup.index:
            st.warning("此 CAS 不在当前描述符表中，暂时无法预测。")
        else:
            st.caption("✓ 已找到此 CAS 对应的四种分子描述符。")

    species = st.selectbox("物种组（Species Group）", list(SPECIES),
                           format_func=lambda n: f"{n} · {SPECIES[n]}")
    max_duration = MAX_DURATION_DAYS[species]
    with st.form("single_prediction"):
        exposure = st.selectbox("暴露类型（Exposure Type）", list(EXPOSURE),
                                format_func=lambda n: f"{n} · {EXPOSURE[n]}")
        medium = st.selectbox("介质类型（Media Type）", list(MEDIA),
                              format_func=lambda n: f"{n} · {MEDIA[n]}")
        endpoint = st.selectbox("终点类型（Endpoint Type）", ENDPOINTS, index=19,
                                format_func=lambda name: f"{name}（编码 {ENDPOINTS.index(name) + 1}）")
        duration = st.number_input("平均观测时长（days）", min_value=0.0,
                                   max_value=max_duration, value=1.0, step=0.01)
        st.caption(f"所选物种组的时长上限：{max_duration:g} days。")
        submitted = st.form_submit_button("预测", type="primary")
    if submitted:
        try:
            log_value, descriptors = predict(model, lookup, cas, species, exposure,
                                             medium, endpoint, duration)
        except ValueError as exc:
            st.error(str(exc))
        else:
            show_result(log_value, descriptors, cas.strip(), species,
                        endpoint, exposure, medium, duration)

with batch_tab:
    st.subheader("CSV 批量预测")
    st.write("下载模板，按行填写 CAS 和五项试验条件，再上传 CSV。每次最多 200 行；错误行在结果中说明原因。")
    example = pd.DataFrame([["50-06-6", 3, 1, 3, "LC50", 1.0]], columns=BATCH_COLUMNS)
    st.download_button("下载 CSV 模板", example.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ppcp_batch_template.csv", mime="text/csv")
    upload = st.file_uploader("上传填好的 CSV 文件", type=["csv"])
    batch_submitted = st.button("批量预测", type="primary")
    if batch_submitted:
        if upload is None:
            st.error("请先上传 CSV 文件。")
        else:
            try:
                frame = pd.read_csv(upload, dtype={"CAS": str}, encoding="utf-8-sig")
                st.session_state["batch_result"] = batch_predict(model, lookup, frame)
            except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
                st.session_state.pop("batch_result", None)
                st.error(f"无法处理文件：{exc}")
    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]
        successful = result["error"].eq("").sum()
        st.write(f"成功 {successful} 行；有错误 {len(result) - successful} 行。")
        st.dataframe(result, hide_index=True, use_container_width=True)
        st.download_button("下载预测结果 CSV", result.to_csv(index=False).encode("utf-8-sig"),
                           file_name="ppcp_predictions.csv", mime="text/csv")
        st.caption("错误行保留在结果中，具体原因见 error 列；这些行未生成预测值。")

with about_tab:
    st.subheader("模型与输入")
    st.write(
        f"当前 CAS 表收录 **{len(lookup)}** 个 CAS。输入 CAS 后，程序从表中读取 AAC、"
        "SpPosA_B(m)、SpMax5_Bh(m) 和 P_VSA_v_3；其余五项为试验条件。"
    )
    st.write("输出是以 mg/L 表示的所选效应浓度的 log10 值；页面同时显示 10 的幂反变换后的 mg/L 浓度。")
    st.write("时长上限：藻类 3 days；两栖动物和鱼类 4 days；甲壳类和昆虫 2 days。模型没有轮虫独立编码。")
    st.write(
        "程序根据提供的训练数据重新训练随机森林。原 RF 的超参数未随数据提供，"
        "因此程序预测不等同于原表 y1。页面未声称对未收录化合物或独立数据集的预测性能。"
    )
    st.write("不同终点（例如 LC50、EC50、NOEC）对应不同效应定义；结果须与终点和试验条件一起解释。")

st.divider()
st.caption("PPCPs 水生效应浓度预测 · CAS 描述符查表 · 随机森林模型")
