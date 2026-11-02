"""Bilingual PPCP aquatic effect-concentration predictor."""

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from model import (
    ENDPOINTS,
    EXPOSURE,
    SPECIES,
    load_data,
    predict,
    train_model,
    valid_cas,
)


ROOT = Path(__file__).resolve().parent
MAX_DURATION_DAYS = {1: 3.0, 2: 4.0, 3: 2.0, 4: 4.0, 5: 2.0}

BATCH_COLUMNS = [
    "CAS",
    "species_group",
    "exposure_type",
    "media_type",
    "endpoint_type",
    "duration_days",
]

LANGUAGE = st.session_state.get("language", "中文")


def t(chinese, english):
    return english if LANGUAGE == "English" else chinese


def clear_results():
    st.session_state.pop("batch_result", None)


def error_text(exc):
    if hasattr(exc, "english"):
        return t(str(exc), exc.english)
    return str(exc)


@st.cache_resource(show_spinner=False)
def get_model():
    train, _, lookup = load_data()
    return train_model(train), lookup


def show_header():
    candidates = [
        ROOT / "asset" / "asset.png",
        ROOT / "assets" / "asset.png",
        ROOT / "assets" / "ppcp_background.jpg",
    ]
    image = next(
        (path for path in candidates if path.is_file()),
        None,
    )

    title = t(
        "PPCPs 水生效应浓度预测",
        "PPCPs Aquatic Effect Concentration Predictor",
    )
    intro = t(
        "输入 CAS 和试验条件，自动查找四种分子描述符并预测相应终点的效应浓度。",
        "Enter a CAS number and test conditions to retrieve four molecular "
        "descriptors and predict the selected endpoint concentration.",
    )

    if image is None:
        st.title(title)
        st.caption(intro)
        return

    mime = (
        "image/png"
        if image.suffix.lower() == ".png"
        else "image/jpeg"
    )

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
            background-image:
                linear-gradient(
                    90deg,
                    rgba(247,250,252,.96) 0%,
                    rgba(247,250,252,.88) 57%,
                    rgba(247,250,252,.30) 100%
                ),
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

        .ppcp-hero h1 {{
            color: #183247;
            font-size: clamp(2rem, 3vw, 2.8rem);
            line-height: 1.2;
            margin: 0 0 14px;
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
            <h1>{title}</h1>
            <p>{intro}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_result(
    log_value, descriptors, cas, species,
    endpoint, exposure, medium, duration,
):
    concentration = 10.0 ** log_value

    with st.container(border=True):
        st.subheader(t("预测结果", "Prediction result"))
        st.metric(
            t("预测效应浓度", "Predicted effect concentration"),
            f"{concentration:.4g} mg/L",
        )
        st.write(
            t("对应的 log10 值：", "Corresponding log10 value: ")
            + f"**{log_value:.4f}**"
        )
        st.caption(
            f"CAS {cas} · {endpoint} · {SPECIES_LABELS[species]} · "
            f"{EXPOSURE_LABELS[exposure]} · {MEDIA_LABELS[medium]} · "
            f"{duration:g} "
            + t("天", "days")
        )

    record = pd.DataFrame([{
        "CAS": cas,
        "species_group": species,
        "exposure_type": exposure,
        "media_type": medium,
        "endpoint_type": endpoint,
        "duration_days": duration,
        "predicted_log10_mg_L": log_value,
        "predicted_mg_L": concentration,
        **descriptors,
    }])

    st.download_button(
        t("下载本次结果 CSV", "Download this result (CSV)"),
        record.to_csv(index=False).encode("utf-8-sig"),
        file_name="ppcp_prediction.csv",
        mime="text/csv",
    )

    with st.expander(
        t(
            "查看自动匹配的四种分子描述符",
            "View the four matched molecular descriptors",
        )
    ):
        st.dataframe(
            pd.DataFrame({
                t("描述符", "Descriptor"): list(descriptors),
                t("数值", "Value"): list(descriptors.values()),
            }),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(t(
        "浓度是模型对所选终点及试验条件的预测，不同终点的效应定义不同。",
        "The concentration is predicted for the selected endpoint and test "
        "conditions. Effect definitions differ between endpoints.",
    ))


def batch_predict(model, lookup, source):
    missing = [
        name for name in BATCH_COLUMNS
        if name not in source.columns
    ]

    if missing:
        raise ValueError(
            t("CSV 缺少列：", "Missing CSV columns: ")
            + ", ".join(missing)
        )

    if source.empty or len(source) > 200:
        raise ValueError(t(
            "每次请上传 1–200 行数据。",
            "Upload 1–200 rows at a time.",
        ))

    rows = []

    for _, item in source.iterrows():
        result = item.to_dict()
        result.update({
            "predicted_log10_mg_L": None,
            "predicted_mg_L": None,
            "error": "",
        })

        try:
            cas = str(item["CAS"]).strip()
            codes = []

            for name in (
                "species_group",
                "exposure_type",
                "media_type",
            ):
                try:
                    numeric = float(item[name])
                except (ValueError, TypeError):
                    raise ValueError(
                        name + t(
                            " 必须是整数编码。",
                            " must be an integer code.",
                        )
                    ) from None

                if not numeric.is_integer():
                    raise ValueError(
                        name + t(
                            " 必须是整数编码。",
                            " must be an integer code.",
                        )
                    )
                codes.append(int(numeric))

            species, exposure, medium = codes

            try:
                duration = float(item["duration_days"])
            except (ValueError, TypeError):
                raise ValueError(t(
                    "duration_days 必须填写数值。",
                    "duration_days must be numeric.",
                )) from None

            if species not in MAX_DURATION_DAYS:
                raise ValueError(t(
                    "物种组编码必须为 1–5。",
                    "Species group code must be 1–5.",
                ))

            maximum = MAX_DURATION_DAYS[species]
            if not 0 <= duration <= maximum:
                raise ValueError(t(
                    f"所选物种组的时长须在 0–{maximum:g} 天之内。",
                    f"Duration must be within 0–{maximum:g} days "
                    "for the selected species group.",
                ))

            endpoint = str(item["endpoint_type"]).strip()

            log_value, _ = predict(
                model, lookup, cas, species,
                exposure, medium, endpoint, duration,
            )

            result["predicted_log10_mg_L"] = log_value
            result["predicted_mg_L"] = 10.0 ** log_value

        except (ValueError, TypeError) as exc:
            result["error"] = error_text(exc)

        rows.append(result)

    return pd.DataFrame(rows)


st.set_page_config(
    page_title="PPCPs | Predictor",
    page_icon="🧪",
    layout="centered",
)

st.radio(
    "语言 / Language",
    ["中文", "English"],
    key="language",
    horizontal=True,
    on_change=clear_results,
)

SPECIES_ZH = {
    1: "藻类",
    2: "两栖动物",
    3: "甲壳类",
    4: "鱼类",
    5: "昆虫",
}
EXPOSURE_ZH = {
    1: "水体暴露（未明确）",
    2: "流水式",
    3: "流动水体",
    4: "脉冲",
    5: "换液式",
    6: "静态",
}
MEDIA_EN = {
    1: "Fresh water",
    2: "Not reported",
    3: "Salt water",
}
MEDIA_ZH = {
    1: "淡水",
    2: "未报告",
    3: "盐水",
}

SPECIES_LABELS = {
    code: t(SPECIES_ZH[code], name)
    for code, name in SPECIES.items()
}
EXPOSURE_LABELS = {
    code: t(EXPOSURE_ZH[code], name)
    for code, name in EXPOSURE.items()
}
MEDIA_LABELS = {
    code: t(MEDIA_ZH[code], name)
    for code, name in MEDIA_EN.items()
}

show_header()

st.info(t(
    "① 选择单条或批量预测 → ② 填写 CAS 与试验条件 → ③ 查看并下载结果",
    "1. Choose single or batch prediction → "
    "2. Enter CAS and test conditions → "
    "3. View and download results",
))

try:
    with st.spinner(t(
        "首次打开时正在训练随机森林模型…",
        "Training the random forest on first use…",
    )):
        model, lookup = get_model()
except (OSError, ValueError) as exc:
    st.error(
        t("无法加载模型或数据：", "Unable to load model or data: ")
        + error_text(exc)
    )
    st.stop()

single_tab, batch_tab, about_tab = st.tabs([
    t("🧪 单条预测", "🧪 Single prediction"),
    t("📄 批量预测", "📄 Batch prediction"),
    t("📖 模型与数据", "📖 Model and data"),
])

with single_tab:
    st.subheader(t("填写预测条件", "Enter prediction conditions"))

    st.caption(t(
        f"当前描述符表可匹配 {len(lookup)} 个 CAS；仅对表内化合物提供预测。",
        f"The descriptor table contains {len(lookup)} CAS numbers; "
        "predictions are available only for listed compounds.",
    ))

    cas = st.text_input(
        t("CAS 号", "CAS number"),
        placeholder=t("例如 50-06-6", "e.g. 50-06-6"),
        key="single_cas",
    )

    if cas.strip():
        if not valid_cas(cas.strip()):
            st.warning(t(
                "CAS 格式或校验位不正确。",
                "Invalid CAS format or check digit.",
            ))
        elif cas.strip() not in lookup.index:
            st.warning(t(
                "此 CAS 不在当前描述符表中，暂时无法预测。",
                "This CAS number is not in the descriptor table; "
                "prediction is unavailable.",
            ))
        else:
            st.caption(t(
                "✓ 已找到此 CAS 对应的四种分子描述符。",
                "✓ Four molecular descriptors found for this CAS number.",
            ))

    species = st.selectbox(
        t("物种组（Species Group）", "Species group"),
        list(SPECIES),
        format_func=lambda n: f"{n} · {SPECIES_LABELS[n]}",
    )
    max_duration = MAX_DURATION_DAYS[species]

    with st.form("single_prediction"):
        exposure = st.selectbox(
            t("暴露类型（Exposure Type）", "Exposure type"),
            list(EXPOSURE),
            format_func=lambda n: f"{n} · {EXPOSURE_LABELS[n]}",
        )

        medium = st.selectbox(
            t("介质类型（Media Type）", "Media type"),
            list(MEDIA_EN),
            format_func=lambda n: f"{n} · {MEDIA_LABELS[n]}",
        )

        endpoint = st.selectbox(
            t("终点类型（Endpoint Type）", "Endpoint type"),
            ENDPOINTS,
            index=19,
            format_func=lambda name: (
                f"{name} "
                + t(
                    f"（编码 {ENDPOINTS.index(name) + 1}）",
                    f"(code {ENDPOINTS.index(name) + 1})",
                )
            ),
        )

        duration = st.number_input(
            t(
                "平均观测时长（天）",
                "Mean observed duration (days)",
            ),
            min_value=0.0,
            max_value=max_duration,
            value=1.0,
            step=0.01,
        )

        st.caption(t(
            f"所选物种组的时长上限：{max_duration:g} 天。",
            f"Duration limit for the selected species group: "
            f"{max_duration:g} days.",
        ))

        submitted = st.form_submit_button(
            t("预测", "Predict"),
            type="primary",
        )

    if submitted:
        try:
            log_value, descriptors = predict(
                model, lookup, cas, species,
                exposure, medium, endpoint, duration,
            )
        except ValueError as exc:
            st.error(error_text(exc))
        else:
            show_result(
                log_value, descriptors, cas.strip(), species,
                endpoint, exposure, medium, duration,
            )

with batch_tab:
    st.subheader(t(
        "CSV 批量预测",
        "Batch prediction from CSV",
    ))

    st.write(t(
        "下载模板，按行填写 CAS 和五项试验条件，再上传 CSV。"
        "每次最多 200 行；错误行在结果中说明原因。",
        "Download the template, enter CAS numbers and five test conditions, "
        "and upload the CSV. Up to 200 rows per upload; "
        "errors are reported for each invalid row.",
    ))

    example = pd.DataFrame(
        [["50-06-6", 3, 1, 3, "LC50", 1.0]],
        columns=BATCH_COLUMNS,
    )

    st.download_button(
        t("下载 CSV 模板", "Download CSV template"),
        example.to_csv(index=False).encode("utf-8-sig"),
        file_name="ppcp_batch_template.csv",
        mime="text/csv",
    )

    upload = st.file_uploader(
        t("上传填好的 CSV 文件", "Upload the completed CSV"),
        type=["csv"],
    )

    batch_submitted = st.button(
        t("批量预测", "Run batch prediction"),
        type="primary",
    )

    if batch_submitted:
        if upload is None:
            st.error(t(
                "请先上传 CSV 文件。",
                "Please upload a CSV file first.",
            ))
        else:
            try:
                frame = pd.read_csv(
                    upload,
                    dtype={"CAS": str},
                    encoding="utf-8-sig",
                )
                st.session_state["batch_result"] = batch_predict(
                    model, lookup, frame,
                )
            except (
                OSError,
                UnicodeError,
                pd.errors.ParserError,
                ValueError,
            ) as exc:
                st.session_state.pop("batch_result", None)
                st.error(
                    t("无法处理文件：", "Unable to process file: ")
                    + error_text(exc)
                )

    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]
        successful = result["error"].eq("").sum()
        failed = len(result) - successful

        st.write(t(
            f"成功 {successful} 行；有错误 {failed} 行。",
            f"Successful rows: {successful}; invalid rows: {failed}.",
        ))

        st.dataframe(
            result,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            t(
                "下载预测结果 CSV",
                "Download prediction results (CSV)",
            ),
            result.to_csv(index=False).encode("utf-8-sig"),
            file_name="ppcp_predictions.csv",
            mime="text/csv",
        )

        st.caption(t(
            "错误行保留在结果中，具体原因见 error 列；"
            "这些行未生成预测值。",
            "Invalid rows are retained with details in the error column; "
            "no predictions are generated for these rows.",
        ))

with about_tab:
    st.subheader(t("模型与输入", "Model and inputs"))

    st.write(t(
        f"当前 CAS 表收录 **{len(lookup)}** 个 CAS。输入 CAS 后，"
        "程序从表中读取 AAC、SpPosA_B(m)、SpMax5_Bh(m) 和 P_VSA_v_3；"
        "其余五项为试验条件。",
        f"The current table contains **{len(lookup)}** CAS numbers. "
        "The app retrieves AAC, SpPosA_B(m), SpMax5_Bh(m) and P_VSA_v_3 "
        "from the table; the other five inputs are test conditions.",
    ))

    st.write(t(
        "输出是以 mg/L 表示的所选效应浓度的 log10 值；"
        "页面同时显示 10 的幂反变换后的 mg/L 浓度。",
        "The model outputs log10 of the selected effect concentration "
        "expressed in mg/L. The page also displays the back-transformed "
        "concentration in mg/L.",
    ))

    st.write(t(
        "时长上限：脊椎动物 4 天，无脊椎动物（轮虫除外）2 天，"
        "轮虫 1 天，藻类 3 天。",
        "Duration limits: vertebrates, 4 days; "
        "invertebrates (excluding rotifers), 2 days; "
        "rotifers, 1 day; algae, 3 days.",
    ))

    st.write(t(
        "程序根据提供的训练数据重新训练随机森林。"
        "原 RF 的超参数未随数据提供，因此程序预测不等同于原表 y1。"
        "页面未声称对未收录化合物或独立数据集的预测性能。",
        "The random forest is refitted using the supplied training data. "
        "Original RF hyperparameters were not supplied, so predictions "
        "are not equivalent to y1 in the original table. "
        "No performance claim is made for unlisted compounds "
        "or independent datasets.",
    ))

    st.write(t(
        "不同终点（例如 LC50、EC50、NOEC）对应不同效应定义；"
        "结果须与终点和试验条件一起解释。",
        "Endpoints such as LC50, EC50 and NOEC have different effect "
        "definitions. Interpret each result together with its endpoint "
        "and test conditions.",
    ))

st.divider()
st.caption(t(
    "PPCPs 水生效应浓度预测 · CAS 描述符查表 · 随机森林模型",
    "PPCPs aquatic effect concentration prediction · "
    "CAS descriptor lookup · Random forest",
))

# END OF APP
