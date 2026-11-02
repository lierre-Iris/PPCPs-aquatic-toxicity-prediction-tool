"""Bilingual PPCP aquatic effect-concentration predictor."""

import base64
import copy
import time
import uuid
from datetime import datetime, timezone
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
    background = "linear-gradient(110deg, #edf7fa, #e4f0f5)"
    image = next((path for path in candidates if path.is_file()), None)
    if image is not None:
        try:
            mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            background = (
                "linear-gradient(90deg, rgba(247,250,252,.97), "
                "rgba(247,250,252,.85), rgba(247,250,252,.45)), "
                f'url("data:{mime};base64,{encoded}")'
            )
        except OSError:
            pass

    title = t("PPCPs 水生效应浓度预测",
              "PPCPs Aquatic Effect Concentration Predictor")
    intro = t("输入 CAS 与试验条件，自动匹配四种分子描述符并预测效应浓度。",
              "Enter a CAS number and test conditions to predict effect concentration.")
    st.markdown(
        f"""
        <style>
        [data-testid="stMainBlockContainer"], .main .block-container {{
            max-width: 1440px;
            padding: 3.5rem 2rem 1.5rem;
        }}
        .ppcp-hero {{
            background-image: {background};
            background-size: cover;
            background-position: center;
            box-sizing: border-box;
            width: 100%;
            min-height: 110px;
            padding: 20px 26px;
            margin: 0 0 6px;
            border: 1px solid #dce9ee;
            border-radius: 14px;
        }}
        .ppcp-hero h1 {{
            color: #183247;
            font-size: clamp(1.35rem, 2vw, 1.85rem);
            line-height: 1.25;
            margin: 0 0 8px;
            padding: 0;
        }}
        .ppcp-hero p {{
            color: #38576a;
            font-size: .95rem;
            line-height: 1.45;
            margin: 0;
        }}
        [data-testid="stMainBlockContainer"] h3 {{
            font-size: 1.25rem;
            padding-top: .2rem;
            padding-bottom: .3rem;
        }}
        [data-testid="stForm"] {{
            padding: 14px 18px;
            border-radius: 12px;
        }}
        @media (max-width: 760px) {{
            [data-testid="stMainBlockContainer"], .main .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}
            .ppcp-hero {{ padding: 18px; }}
        }}
        </style>
        <div class="ppcp-hero"><h1>{title}</h1><p>{intro}</p></div>
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


TASK_TTL_SECONDS = 24 * 60 * 60


def prune_tasks(now=None):
    """Expire this session's tasks. Previewing never extends retention."""
    now = time.time() if now is None else now
    tasks = st.session_state.setdefault("tasks", {})
    expired = [key for key, task in tasks.items() if task["expires_at"] <= now]
    for key in expired:
        del tasks[key]
    return len(expired)


def add_task(kind, payload, submitted_at, source_name=""):
    prune_tasks()
    task_id = uuid.uuid4().hex
    st.session_state["tasks"][task_id] = {
        "kind": kind,
        "created_at": submitted_at,
        "expires_at": submitted_at + TASK_TTL_SECONDS,
        "source_name": source_name,
        "payload": copy.deepcopy(payload),
    }
    st.session_state["task_preview"] = task_id
    return task_id


def delete_tasks(task_ids):
    tasks = st.session_state.setdefault("tasks", {})
    count = 0
    for task_id in set(task_ids):
        if task_id in tasks:
            del tasks[task_id]
            count += 1
    return count


def task_time(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@st.fragment(run_every="60s")
def show_task_history():
    # All saved result displays live inside this fragment, so cleanup also
    # removes expired previews and download buttons from the current page.
    expired_count = prune_tasks()
    st.divider()
    st.subheader(t("我的任务", "My tasks"))
    st.caption(t(
        "仅显示当前浏览器会话的任务。任务自提交起保留至多 24 小时；"
        "页面活跃时每分钟检查过期任务。会话结束或服务重启可能提前清空记录。"
        "请及时下载需要保存的结果。时间统一显示为 UTC。",
        "Tasks belong to this browser session and expire 24 hours after submission. "
        "Expiry is checked every minute while the session is active. "
        "Ending the session or restarting the service may clear tasks earlier. "
        "Download results you wish to keep. All timestamps are in UTC.",
    ))
    if expired_count:
        st.info(t(f"已自动清理 {expired_count} 个过期任务。",
                  f"Automatically removed {expired_count} expired task(s)."))
    notice = st.session_state.pop("task_deleted_notice", 0)
    if notice:
        st.success(t(f"已删除 {notice} 个任务及其保存的结果。",
                     f"Deleted {notice} task(s) and their saved results."))
    tasks = st.session_state["tasks"]
    if not tasks:
        st.info(t("暂无任务。提交预测后，结果将出现在这里。",
                  "No tasks yet. Submit a prediction to see its results here."))
        return

    task_ids = sorted(tasks, key=lambda key: tasks[key]["created_at"], reverse=True)
    labels = {}
    records = []
    for task_id in task_ids:
        task = tasks[task_id]
        kind = t("单条", "Single") if task["kind"] == "single" else t("批量", "Batch")
        summary = (task["payload"]["cas"] if task["kind"] == "single"
                   else task["source_name"])
        labels[task_id] = f"{task_id[:8]} · {kind} · {summary}"
        records.append({
            t("任务编号", "Task ID"): task_id[:8],
            t("类型", "Type"): kind,
            t("CAS / 文件", "CAS / File"): summary,
            t("记录数", "Rows"): 1 if task["kind"] == "single" else len(task["payload"]),
            t("提交时间", "Submitted"): task_time(task["created_at"]),
            t("到期时间", "Expires"): task_time(task["expires_at"]),
        })
    st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)

    # Repair selections before instantiating widgets after deletion or expiry.
    if st.session_state.get("task_preview") not in tasks:
        st.session_state["task_preview"] = task_ids[0]
    st.session_state["tasks_to_delete"] = [
        key for key in st.session_state.get("tasks_to_delete", []) if key in tasks
    ]
    selected = st.selectbox(t("选择任务预览", "Select a task to preview"), task_ids,
                            format_func=lambda key: labels[key], key="task_preview")
    to_delete = st.multiselect(
        t("选择要删除的任务（可多选）", "Select tasks to delete (multiple allowed)"),
        task_ids, format_func=lambda key: labels[key], key="tasks_to_delete",
    )
    if st.button(t("删除所选任务", "Delete selected tasks"),
                 key="delete_selected_tasks", disabled=not to_delete):
        st.session_state["task_deleted_notice"] = delete_tasks(to_delete)
        st.rerun()
    st.caption(t("删除后无法恢复；已下载到本地的文件不受影响。",
                 "Deletion cannot be undone. Files already downloaded are unaffected."))

    # Recheck at the moment of preview, in addition to the periodic sweep.
    prune_tasks()
    if selected not in tasks:
        st.rerun()
    task = tasks[selected]
    st.caption(labels[selected] + " · " + t("到期：", "Expires: ")
               + task_time(task["expires_at"]))
    if task["kind"] == "single":
        show_result(**task["payload"])
    else:
        result = task["payload"]
        successful = int(result["error"].eq("").sum())
        failed = len(result) - successful
        st.write(t(f"成功 {successful} 行；有错误 {failed} 行。",
                   f"Successful rows: {successful}; invalid rows: {failed}."))
        st.dataframe(result, hide_index=True, use_container_width=True)
        st.download_button(
            t("下载预测结果 CSV", "Download predictions CSV"),
            result.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"ppcp_predictions_{selected[:8]}.csv", mime="text/csv",
            key=f"batch_download_{selected}",
        )
        st.caption(t(
            "错误行没有预测值，原因见 error 列。错误说明保留提交时的语言。",
            "Invalid rows have no prediction; see the error column. "
            "Error messages keep the language used at submission.",
        ))



st.set_page_config(
    page_title="PPCPs | Predictor", page_icon="🧪",
    layout="wide", initial_sidebar_state="expanded",
)

# Clear obsolete one-result storage; all new results live in tasks only.
st.session_state.pop("single_result", None)
st.session_state.pop("batch_result", None)
prune_tasks()

with st.sidebar:
    st.markdown("## 🧪 PPCPs")
    LANGUAGE = st.radio("语言 / Language", ["中文", "English"], key="language")

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


pages = {
    "single": t("🧪 单条预测", "🧪 Single prediction"),
    "batch": t("📄 批量预测", "📄 Batch prediction"),
    "about": t("📖 模型与数据", "📖 Model and data"),
}

with st.sidebar:
    st.divider()
    page = st.radio(
        t("功能导航", "Navigation"), list(pages),
        format_func=lambda code: pages[code], key="navigation",
    )
    st.divider()
    st.caption(t("使用步骤", "Quick start"))
    st.write(t(
        "① 选择预测方式\n\n② 填写 CAS 和试验条件\n\n③ 查看并下载结果",
        "① Choose a prediction mode\n\n② Enter CAS and test conditions"
        "\n\n③ View and download results",
    ))

show_header()

try:
    with st.spinner(t("正在加载数据和模型，首次打开需要训练…",
                      "Loading data and model; training is required on first use…")):
        model, lookup = get_model()
except (OSError, ValueError) as exc:
    st.error(t("无法加载模型或数据：", "Unable to load model or data: ")
             + error_text(exc))
    st.stop()

if page == "single":
    st.subheader(t("填写预测条件", "Enter prediction conditions"))
    st.caption(t(
        f"描述符表包含 {len(lookup)} 个 CAS；仅支持表内化合物。",
        f"The descriptor table contains {len(lookup)} CAS numbers; "
        "only listed compounds are supported.",
    ))

    left, right = st.columns(2, gap="medium")
    with left:
        cas = st.text_input(t("CAS 号", "CAS number"),
                            placeholder=t("例如 50-06-6", "e.g. 50-06-6"),
                            key="single_cas")
    with right:
        species = st.selectbox(
            t("物种组", "Species group"), list(SPECIES),
            format_func=lambda n: f"{n} · {SPECIES_LABELS[n]}",
            key="single_species",
        )
    max_duration = MAX_DURATION_DAYS[species]

    # Species is outside the form so its duration limit updates immediately.
    with st.form("single_prediction"):
        left, right = st.columns(2, gap="medium")
        with left:
            exposure = st.selectbox(
                t("暴露类型", "Exposure type"), list(EXPOSURE),
                format_func=lambda n: f"{n} · {EXPOSURE_LABELS[n]}",
                key="single_exposure",
            )
        with right:
            medium = st.selectbox(
                t("介质类型", "Media type"), list(MEDIA_EN),
                format_func=lambda n: f"{n} · {MEDIA_LABELS[n]}",
                key="single_medium",
            )
        left, right = st.columns(2, gap="medium")
        with left:
            endpoint = st.selectbox(
                t("终点类型", "Endpoint type"), ENDPOINTS,
                index=ENDPOINTS.index("LC50") if "LC50" in ENDPOINTS else 0,
                format_func=lambda name: (
                    f"{name} · " + t("编码 ", "Code ")
                    + str(ENDPOINTS.index(name) + 1)
                ),
                key="single_endpoint",
            )
        with right:
            duration = st.number_input(
                t("平均观测时长（天）", "Mean observed duration (days)"),
                min_value=0.0, max_value=float(max_duration),
                value=1.0, step=0.01, key=f"duration_{species}",
            )
        st.caption(t(
            f"当前物种组：{SPECIES_LABELS[species]}；允许时长：0–{max_duration:g} 天。",
            f"Selected group: {SPECIES_LABELS[species]}; "
            f"allowed duration: 0–{max_duration:g} days.",
        ))
        submitted = st.form_submit_button(
            t("开始预测", "Predict"), type="primary", use_container_width=True,
        )

    clean_cas = cas.strip()
    if clean_cas and not submitted:
        if not valid_cas(clean_cas):
            st.warning(t("CAS 格式或校验位不正确。", "Invalid CAS format or check digit."))
        elif clean_cas not in lookup.index:
            st.warning(t("此 CAS 不在当前描述符表中。",
                         "This CAS number is not in the descriptor table."))
        else:
            st.caption(t("✓ 已匹配四种分子描述符。", "✓ Four molecular descriptors matched."))

    if submitted:
        submitted_at = time.time()
        try:
            log_value, descriptors = predict(
                model, lookup, clean_cas, species, exposure, medium, endpoint, duration,
            )
        except ValueError as exc:
            st.error(error_text(exc))
        else:
            result_payload = dict(
                log_value=log_value, descriptors=descriptors, cas=clean_cas,
                species=species, endpoint=endpoint, exposure=exposure,
                medium=medium, duration=duration,
            )

            add_task("single", result_payload, submitted_at)
            st.success(t("预测已完成，请在下方“我的任务”中查看。",
                         "Prediction complete. View it in My tasks below."))

elif page == "batch":
    st.subheader(t("CSV 批量预测", "Batch prediction from CSV"))
    st.caption(t(
        "下载模板，填写 CAS 和五项试验条件。每次支持 1–200 行。",
        "Download the template and enter CAS numbers and five test conditions. "
        "Upload 1–200 rows per batch.",
    ))
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(t("#### 1. 下载模板", "#### 1. Download template"))
        example = pd.DataFrame(
            [["50-06-6", 3, 1, 3, "LC50", 1.0]], columns=BATCH_COLUMNS,
        )
        st.download_button(
            t("下载 CSV 模板", "Download CSV template"),
            example.to_csv(index=False).encode("utf-8-sig"),
            file_name="ppcp_batch_template.csv", mime="text/csv",
            use_container_width=True,
        )
        st.caption(t(
            "保留英文列名；分类变量填写数字编码，终点填写 LC50 等原始名称。",
            "Keep the English column names. Use numeric category codes "
            "and endpoint names such as LC50.",
        ))
        with st.expander(t("查看分类编码", "View category codes")):
            for heading, mapping in (
                (t("物种组", "Species group"), SPECIES_LABELS),
                (t("暴露类型", "Exposure type"), EXPOSURE_LABELS),
                (t("介质类型", "Media type"), MEDIA_LABELS),
            ):
                st.write(heading)
                st.write(" · ".join(f"{code}: {label}" for code, label in mapping.items()))
            st.write(t("终点类型", "Endpoint types"))
            st.write(", ".join(ENDPOINTS))
    with right:
        st.markdown(t("#### 2. 上传并预测", "#### 2. Upload and predict"))
        upload = st.file_uploader(t("上传填好的 CSV", "Upload completed CSV"),
                                  type=["csv"], key="batch_upload")
        batch_submitted = st.button(t("开始批量预测", "Run batch prediction"),
                                    type="primary", use_container_width=True)
    if batch_submitted:
        submitted_at = time.time()
        if upload is None:
            st.error(t("请先上传 CSV 文件。", "Please upload a CSV file first."))
        else:
            try:
                upload.seek(0)
                frame = pd.read_csv(upload, dtype={"CAS": str}, encoding="utf-8-sig")
                result = batch_predict(model, lookup, frame)
                add_task("batch", result, submitted_at, source_name=upload.name)
                st.success(t("批量预测已完成，请在下方“我的任务”中查看。",
                             "Batch complete. View it in My tasks below."))
            except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
                st.error(t("无法处理文件：", "Unable to process file: ") + error_text(exc))

else:
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

    st.caption(t(
        "当前模型没有轮虫独立编码；界面按现有五类物种限制观测时长。",
        "The current model has no separate rotifer code; duration limits "
        "follow the five available species groups.",
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

show_task_history()

st.divider()
st.caption(t(
    "PPCPs 水生效应浓度预测 · CAS 描述符查表 · 随机森林模型",
    "PPCPs aquatic effect concentration prediction · "
    "CAS descriptor lookup · Random forest",
))

# END OF APP
