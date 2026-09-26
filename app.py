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

st.subheader("单条预测")
st.write("输入化合物 CAS 号并选择试验条件，预测所选终点对应的效应浓度。")
with st.expander("使用前先了解预测范围"):
    st.write(
        "CAS 必须存在于随附的描述符表中；程序不会根据新的 CAS 自动计算描述符。"
        "物种组、暴露类型、介质类型和终点类型按训练数据的编码选择。"
    )
    st.write(
        "结果是所选效应终点的浓度，不同终点的效应含义不同。"
        "预测浓度越高，达到该指定效应所需的浓度越高。"
    )

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
max_duration = MAX_DURATION_DAYS[species]

with st.form("prediction"):
    cas = st.text_input("CAS 号", placeholder="例如 50-06-6")
    exposure = st.selectbox(
        "暴露类型（Exposure Type）", list(EXPOSURE),
        format_func=lambda n: f"{n} · {EXPOSURE[n]}",
    )
    medium = st.selectbox(
        "介质类型（Media Type）", list(MEDIA),
        format_func=lambda n: f"{n} · {MEDIA[n]}",
    )
    endpoint = st.selectbox(
        "终点类型（Endpoint Type）", ENDPOINTS, index=19,
        format_func=lambda item: f"{item}（编码 {ENDPOINTS.index(item) + 1}）",
    )
    duration = st.number_input(
        "平均观测时长（Observed Duration Mean，days）",
        min_value=0.0, max_value=max_duration, value=1.0, step=0.01,
    )
    st.caption(f"当前物种组的平均观测时长上限：{max_duration:g} days。")
    submitted = st.form_submit_button("预测", type="primary")

if submitted:
    try:
        log_value, descriptors = predict(
            model, lookup, cas, species, exposure, medium, endpoint, duration
        )
    except ValueError as exc:
        st.error(str(exc))
    else:
        concentration_mg_l = 10.0 ** log_value
        with st.container(border=True):
            st.subheader("预测结果")
            st.metric("预测效应浓度", f"{concentration_mg_l:.4g} mg/L")
            st.write(f"对应的 log10 值：**{log_value:.4f}**")
            st.caption(
                f"CAS：{cas.strip()}；终点：{endpoint}；物种组：{SPECIES[species]}；"
                f"暴露类型：{EXPOSURE[exposure]}；介质：{MEDIA[medium]}；"
                f"观测时长：{duration:g} days。"
            )

        st.download_button(
            "下载本次预测结果（CSV）",
            data=result_csv(cas, species, exposure, medium, endpoint, duration,
                            log_value, concentration_mg_l, descriptors),
            file_name="ppcp_prediction.csv",
            mime="text/csv",
        )
        with st.expander("查看 CAS 自动匹配的分子描述符"):
            st.dataframe(
                {"描述符": list(descriptors), "数值": list(descriptors.values())},
                hide_index=True, use_container_width=True,
            )

st.divider()
with st.expander("模型与数据说明"):
    st.write(
        "程序从 CAS 对照表读取 AAC、SpPosA_B(m)、SpMax5_Bh(m)、P_VSA_v_3；"
        "再结合物种组、暴露类型、介质类型、终点类型与平均观测时长，"
        "用随机森林预测效应浓度（mg/L）的 log10 值。页面显示反变换后的 mg/L 浓度。"
    )
    st.write(
        "时长输入范围按物种组设置：藻类不超过 3 days；两栖动物和鱼类不超过 4 days；"
        "甲壳类和昆虫不超过 2 days。界面没有单独的轮虫物种组。"
    )
    st.write(
        "原 RF 的超参数未随附件提供；本程序根据随附训练数据重新拟合随机森林，"
        "因此输出不等同于原表 y1。不同终点的结果应结合对应效应定义及试验条件解释。"
    )

st.caption("PPCPs 水生效应浓度预测 · CAS 查表 · 随机森林")
