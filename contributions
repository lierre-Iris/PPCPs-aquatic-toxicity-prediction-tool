"""Experimental data contributions saved to a private GitHub repository."""

import base64
import csv
import hashlib
import io
import json
import math
import re
import uuid

from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from model import ENDPOINTS, EXPOSURE, SPECIES, valid_cas


MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 1000

LIMITS = {
    1: 3,
    2: 4,
    3: 2,
    4: 4,
    5: 2,
}

COLUMNS = [
    "CAS",
    "species_group",
    "exposure_type",
    "media_type",
    "endpoint_type",
    "duration_days",
    "observed_mg_L",
    "reference",
]


def tr(chinese, english):
    language = st.session_state.get("language", "中文")
    return english if language == "English" else chinese


def validate_csv(raw):
    """Validate uploaded experimental data before submission."""
    if len(raw) > MAX_BYTES:
        raise ValueError(
            tr(
                "文件不能超过 2 MB。",
                "The file must not exceed 2 MB.",
            )
        )

    try:
        text = raw.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text)))

        if len(set(header)) != len(header):
            raise ValueError(
                tr(
                    "CSV 中有重复列名。",
                    "Duplicate CSV column names.",
                )
            )

        frame = pd.read_csv(
            io.StringIO(text),
            dtype=str,
            keep_default_na=False,
            nrows=MAX_ROWS + 1,
        )

    except (
        UnicodeError,
        StopIteration,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ):
        raise ValueError(
            tr(
                "请上传 UTF-8 编码的标准 CSV 文件。",
                "Upload a standard CSV file encoded as UTF-8.",
            )
        ) from None

    missing = [
        name for name in COLUMNS
        if name not in frame.columns
    ]

    if missing:
        raise ValueError(
            tr("缺少列：", "Missing columns: ")
            + ", ".join(missing)
        )

    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(
            tr(
                "每次提交 1–1000 行。",
                "Submit 1–1000 rows at a time.",
            )
        )

    records = []
    errors = []

    for index, item in frame.iterrows():
        row = {
            str(key): str(value).strip()
            for key, value in item.items()
        }

        try:
            if not valid_cas(row["CAS"]):
                raise ValueError(
                    tr(
                        "CAS 格式或校验位不正确。",
                        "Invalid CAS format or check digit.",
                    )
                )

            for name, allowed in (
                ("species_group", SPECIES),
                ("exposure_type", EXPOSURE),
                ("media_type", (1, 2, 3)),
            ):
                value = float(row[name])

                if (
                    not math.isfinite(value)
                    or not value.is_integer()
                    or int(value) not in allowed
                ):
                    raise ValueError(
                        tr(
                            "分类编码不正确：",
                            "Invalid category code: ",
                        ) + name
                    )

                row[name] = int(value)

            if row["endpoint_type"] not in ENDPOINTS:
                raise ValueError(
                    tr(
                        "终点名称不在支持列表中。",
                        "Unsupported endpoint name.",
                    )
                )

            row["duration_days"] = float(row["duration_days"])

            if not (
                0 < row["duration_days"]
                <= LIMITS[row["species_group"]]
            ):
                raise ValueError(
                    tr(
                        "观测时长应大于 0，且不超过该物种组的急性时长上限。",
                        "Duration must be positive and within "
                        "the group's acute duration limit.",
                    )
                )

            row["observed_mg_L"] = float(row["observed_mg_L"])

            if (
                not math.isfinite(row["observed_mg_L"])
                or row["observed_mg_L"] <= 0
            ):
                raise ValueError(
                    tr(
                        "实验浓度必须为有限正数，单位为 mg/L。",
                        "Observed concentration must be finite "
                        "and positive, in mg/L.",
                    )
                )

            if not row["reference"]:
                raise ValueError(
                    tr(
                        "请填写文献或实验来源。",
                        "A literature or experiment reference is required.",
                    )
                )

        except (ValueError, OverflowError) as exc:
            errors.append({
                tr("CSV 行号", "CSV line"): index + 2,
                tr("问题", "Issue"): str(exc),
            })

        records.append(row)

    return records, errors


def storage_config():
    """Read configuration from Streamlit Secrets."""
    try:
        settings = st.secrets["data_collection"]

        config = {
            key: str(settings[key]).strip()
            for key in ("token", "repository", "branch")
        }

    except (KeyError, FileNotFoundError):
        return None

    repository_pattern = r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"

    if (
        not all(config.values())
        or not re.fullmatch(
            repository_pattern,
            config["repository"],
        )
    ):
        return None

    return config


def github_request(config, method, suffix, payload=None):
    """Call GitHub without exposing credentials or private response details."""
    url = (
        "https://api.github.com/repos/"
        + config["repository"]
        + suffix
    )

    body = (
        None
        if payload is None
        else json.dumps(payload).encode("utf-8")
    )

    request = Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": "Bearer " + config["token"],
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "PPCP-data-contributions",
        },
    )

    try:
        with urlopen(request, timeout=25) as response:
            return response.status, json.load(response)

    except HTTPError as exc:
        return exc.code, {}

    except (URLError, TimeoutError, OSError, ValueError):
        return 0, {}


def save_submission(config, record):
    """Save one submission, safely reusing its ID when retrying."""
    status, repository = github_request(
        config,
        "GET",
        "",
    )

    if (
        status != 200
        or repository.get("private") is not True
    ):
        raise RuntimeError(
            tr(
                "无法确认私有收集仓库，请联系管理员检查配置。",
                "Cannot verify the private collection repository. "
                "Contact the administrator.",
            )
        )

    raw = json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")

    path = (
        "submissions/"
        + record["submitted_at"][:10]
        + "/"
        + record["submission_id"]
        + ".json"
    )

    suffix = "/contents/" + quote(path, safe="/")

    expected_sha = hashlib.sha1(
        b"blob "
        + str(len(raw)).encode()
        + b"\0"
        + raw
    ).hexdigest()

    def already_saved():
        check_status, existing = github_request(
            config,
            "GET",
            suffix
            + "?ref="
            + quote(config["branch"], safe=""),
        )

        return (
            check_status == 200
            and existing.get("sha") == expected_sha
        )

    if already_saved():
        return

    status, _ = github_request(
        config,
        "PUT",
        suffix,
        {
            "message": (
                "Receive experimental dataset "
                + record["submission_id"]
            ),
            "content": base64.b64encode(raw).decode("ascii"),
            "branch": config["branch"],
        },
    )

    if status == 201 or already_saved():
        return

    raise RuntimeError(
        tr(
            "尚未确认保存成功，请稍后重试。"
            "如持续失败，请联系管理员。重试会复用本次提交编号。",
            "Storage has not been confirmed. Retry later or contact "
            "the administrator. Retries reuse this submission ID.",
        )
    )


@st.dialog("感谢提交 / Thank you", width="small")
def thank_you(submission_id, count):
    """Display a thank-you dialog only after storage is confirmed."""
    st.success(
        tr(
            "感谢您为 PPCPs 水生毒性研究贡献实验数据！",
            "Thank you for contributing experimental data "
            "to PPCP aquatic toxicity research!",
        )
    )

    st.write(
        tr(
            f"您的 {count} 条记录已保存，"
            "将在审核后用于评估后续模型优化。",
            f"Your {count} records have been saved for review "
            "and potential future model improvement.",
        )
    )

    st.caption(
        tr("提交编号：", "Submission ID: ")
        + submission_id
    )

    if st.button(
        tr("完成", "Done"),
        key="contribution_thanks_done",
    ):
        st.rerun()


def render_contribution_portal():
    """Render the bilingual experimental-data submission section."""
    st.subheader(
        tr(
            "实验数据提交",
            "Contribute experimental data",
        )
    )

    st.write(
        tr(
            "欢迎提交有来源可追溯的实验效应浓度数据，帮助扩充数据库。"
            "数据将保存至研究团队的私有仓库，经人工审核、去重和"
            "质量评估后，再考虑用于模型更新。",
            "Contribute traceable experimental effect-concentration "
            "data to expand the database. Data are stored in the "
            "research team's private repository for manual review, "
            "deduplication and quality assessment before any model update.",
        )
    )

    st.caption(
        tr(
            "实验数据长期收集，不受预测任务的 24 小时清理规则影响。"
            "目前接收五类水生生物、24 类终点及其急性时长范围内的数据；"
            "CAS 无需已在预测查表中。",
            "Contributions are retained separately from the 24-hour "
            "prediction task history. The five aquatic groups, supported "
            "endpoints and acute duration limits apply. CAS numbers need "
            "not already exist in the prediction lookup table.",
        )
    )

    template = (
        ",".join(COLUMNS) + "\n"
    ).encode("utf-8-sig")

    st.download_button(
        tr(
            "下载实验数据模板",
            "Download experimental data template",
        ),
        template,
        file_name="experimental_data_template.csv",
        mime="text/csv",
        key="contribution_template",
    )

    with st.expander(
        tr(
            "填写说明与分类编码",
            "Instructions and category codes",
        )
    ):
        st.write(
            tr(
                "observed_mg_L：实测效应浓度，单位 mg/L，"
                "填写原始浓度而非 log10 值。",
                "observed_mg_L: experimentally measured effect "
                "concentration in mg/L, not log10.",
            )
        )

        st.write(
            tr(
                "reference：DOI、论文题目或可追溯的实验记录编号。"
                "每行均需填写。",
                "reference: DOI, paper title or traceable experiment ID. "
                "Required for every row.",
            )
        )

        st.write(
            tr(
                "duration_days：观测时长，单位为天。"
                "请勿提交预测值或将检出限当作精确实验值。",
                "duration_days: exposure duration in days. "
                "Do not submit predictions or detection limits "
                "as exact measurements.",
            )
        )

        st.write(
            tr(
                "可另加 compound_name、species_name、notes 等列，"
                "系统会一并保存。",
                "Optional columns such as compound_name, species_name "
                "and notes will also be retained.",
            )
        )

        st.write(
            "species_group: "
            + "; ".join(
                f"{code} = {label}"
                for code, label in SPECIES.items()
            )
        )

        st.write(
            "exposure_type: "
            + "; ".join(
                f"{code} = {label}"
                for code, label in EXPOSURE.items()
            )
        )

        st.write(
            "media_type: 1 = Fresh water; "
            "2 = Not reported; 3 = Salt water"
        )

        st.write(
            "endpoint_type: " + ", ".join(ENDPOINTS)
        )

    config = storage_config()

    if config is None:
        st.info(
            tr(
                "数据提交服务尚未启用。管理员完成存储配置后即可提交；"
                "模板仍可下载。",
                "Submissions are not enabled yet. The administrator "
                "must configure storage. The template is available.",
            )
        )

    upload = st.file_uploader(
        tr(
            "上传实验数据 CSV（UTF-8，最大 2 MB，最多 1000 行）",
            "Upload experimental CSV (UTF-8, max 2 MB, 1000 rows)",
        ),
        type=["csv"],
        key="contribution_upload",
    )

    records = None

    if upload is not None:
        try:
            if upload.size > MAX_BYTES:
                raise ValueError(
                    tr(
                        "文件不能超过 2 MB。",
                        "File must not exceed 2 MB.",
                    )
                )

            records, errors = validate_csv(upload.getvalue())

            st.dataframe(
                pd.DataFrame(records).head(20),
                hide_index=True,
                use_container_width=True,
            )

            if errors:
                st.error(
                    tr(
                        "请修正以下问题后重新上传，当前文件不会被提交。",
                        "Correct these issues and reupload. "
                        "This file cannot be submitted.",
                    )
                )

                st.dataframe(
                    pd.DataFrame(errors),
                    hide_index=True,
                    use_container_width=True,
                )

                records = None

            else:
                st.caption(
                    tr(
                        f"校验通过，共 {len(records)} 行；"
                        "上方预览前 20 行。",
                        f"Validated {len(records)} rows; "
                        "showing the first 20.",
                    )
                )

        except ValueError as exc:
            st.error(str(exc))

    with st.form("contribution_form"):
        name = st.text_input(
            tr(
                "提交人或团队名称（必填）",
                "Contributor or team (required)",
            ),
            max_chars=150,
        )

        email = st.text_input(
            tr(
                "联系邮箱（选填，用于数据核实）",
                "Email (optional, for data clarification)",
            ),
            max_chars=254,
        )

        notes = st.text_area(
            tr(
                "补充说明（选填）",
                "Additional notes (optional)",
            ),
            max_chars=3000,
        )

        consent = st.checkbox(
            tr(
                "我确认有权提交这些实验数据，并同意研究团队保存、"
                "审核及用于后续模型研究。",
                "I have the right to submit these experimental data "
                "and consent to their storage, review and use "
                "in future model research.",
            )
        )

        submit = st.form_submit_button(
            tr(
                "提交实验数据",
                "Submit experimental data",
            ),
            type="primary",
            disabled=config is None or records is None,
        )

    if submit:
        if not name.strip() or not consent:
            st.error(
                tr(
                    "请填写提交人或团队名称，并勾选数据使用授权。",
                    "Enter a contributor or team name and confirm consent.",
                )
            )
            return

        if (
            email.strip()
            and not re.fullmatch(
                r"[^\s@]+@[^\s@]+\.[^\s@]+",
                email.strip(),
            )
        ):
            st.error(
                tr(
                    "请填写有效邮箱，或留空。",
                    "Enter a valid email or leave it blank.",
                )
            )
            return

        content = {
            "records": records,
            "contributor": name.strip(),
            "email": email.strip(),
            "notes": notes.strip(),
            "consent": True,
            "consent_version": "1",
        }

        fingerprint = hashlib.sha256(
            json.dumps(
                content,
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

        attempts = st.session_state.setdefault(
            "contribution_attempts",
            {},
        )

        if fingerprint not in attempts:
            attempts[fingerprint] = {
                **content,
                "submission_id": uuid.uuid4().hex,
                "submitted_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "status": "pending_review",
                "schema_version": 1,
                "original_filename": upload.name,
            }

        record = attempts[fingerprint]

        try:
            with st.spinner(
                tr(
                    "正在保存数据…",
                    "Saving data…",
                )
            ):
                save_submission(config, record)

        except RuntimeError as exc:
            st.error(str(exc))

        else:
            st.session_state["contribution_receipt"] = (
                record["submission_id"]
            )

            thank_you(
                record["submission_id"],
                len(records),
            )

    if st.session_state.get("contribution_receipt"):
        st.caption(
            tr(
                "最近一次已确认保存的提交编号：",
                "Last confirmed submission ID: ",
            )
            + st.session_state["contribution_receipt"]
        )


# END OF CONTRIBUTIONS
