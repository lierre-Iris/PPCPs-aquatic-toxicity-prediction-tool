# PPCPs 水生效应浓度预测小程序

本项目使用用户提供的 `RF建模.xlsx` 中 `traindata` 重新训练随机森林，使用 `四种分子描述符汇总(1).xlsx` 建立 **CAS → 四种分子描述符** 的查找表。界面输入 CAS、物种组、暴露类型、介质类型、终点类型、平均观测时长，共六项；模型实际使用九项特征。输入 CAS 时无需手工填写四个描述符。

## 直接部署到 GitHub 和 Streamlit

1. 下载本项目 ZIP 并解压。打开解压后名为 `ppcp_rf_predictor` 的文件夹，确认能看到 `app.py`、`model.py`、`requirements.txt`、`README.md` 和 `data/`。
2. 登录 GitHub，选择 **New repository**，创建一个仓库，例如 `ppcp-aquatic-predictor`。用 **Add file → Upload files** 上传上述文件和 `data` 文件夹中的三个 CSV 文件。上传时应使 `app.py` 位于仓库根目录；可以直接将文件夹中的内容整体拖到 GitHub 上传区。提交（**Commit changes**）。如果仓库公开，CSV 数据也会公开，请先确认可公开。
3. 打开 [Streamlit Community Cloud](https://share.streamlit.io/)，用 GitHub 登录，点击 **Create app**，选中刚创建的仓库，分支选 `main`（或你的实际分支），**Main file path** 填 `app.py`，点击 **Deploy**。首次启动需安装依赖并训练模型，稍等即可使用。
4. 在网页依次填写六项信息，点 **预测**。例如 CAS `50-06-6` 在描述符表内。没有收录的 CAS 会得到明确提示，不会拿其他化合物的描述符代替。

详细部署界面会随平台更新；官方步骤见 [Streamlit 部署文档](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)。本项目没有直接上传到你的 GitHub 账户，所以还需要你完成第 2、3 步。

### 本机运行（可选）

需要 Python 3.10 或更新版本。在项目文件夹内运行：

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

运行随附的逻辑检查：`python -m unittest discover -s tests -v`。

## CAS 自动填入描述符的具体实现

`data/cas_descriptors.csv` 每行保存一个 CAS 与 `AAC`、`SpPosA_B(m)`、`SpMax5_Bh(m)`、`P_VSA_v_3` 的对应关系。用户输入 CAS 后，`model.py` 首先去除首尾空格、校验 CAS 校验位，再按 CAS 精确查找四列数值；找不到则停止预测。程序按 `x1…x9` 的训练列顺序拼成一行，交给 RF 预测。代码**不会根据 CAS 号计算分子结构**，也**不会联网搜索未收录化合物**。目前查找表含 103 个不同 CAS。

要增加新化合物，请在 `data/cas_descriptors.csv` 增加一行真实 CAS 和用**相同描述符计算流程**得到的四个数值，提交 GitHub 后重新部署。即便新增 CAS，若其描述符落在训练数据覆盖范围之外，仍不能据此保证预测可靠。若模型训练数据发生变化，请用相同表头更新 `data/train.csv`；测试表 `data/test.csv` 仅用于离线核查。

## 列映射和输入定义

| 输入 | 数据列 | 界面处理 |
| --- | --- | --- |
| Species Group | `x1` | 1 Algae；2 Amphibians；3 Crustaceans；4 Fish；5 Insects |
| Exposure Type | `x2` | 1 Aquatuc（沿用此前提供的原始拼写，须核对）；2 Flow-through；3 Lotic；4 Pulse；5 Renewal；6 Static |
| Media Type | `x3` | 1、2、3；附件没有给出名称，请对照原始数据字典选择 |
| Endpoint Type | `x4` | 按下方列表由 1 到 24 编码 |
| Observed Duration Mean | `x5` | 原始建模数值；训练范围 0–4；附件未注明时间单位 |
| CAS 自动查找 | `x6`–`x9` | 依次是 AAC、SpPosA_B(m)、SpMax5_Bh(m)、P_VSA_v_3 |
| 目标 | `y` | log 转换后的效应浓度；`y1` 不参与训练 |

终点编码依次为：1 BMC05；2 BMC20；3 ECO；4 EC05；5 EC10；6 EC100；7 EC16；8 EC20；9 EC25；10 EC50；11 EC80；12 EC84；13 EC90；14 ER50；15 ET50；16 IC10；17 IC20；18 IC25；19 IC50；20 LC50；21 LD50；22 LOEC；23 NOEC；24 NR-LETH。

**输出目前仅为 log 尺度**。附件无法证明 `y` 的对数底数及原始浓度单位，不能擅自标成 mg/L 或计算 `10^y`。确定原始转换公式后，可在 `app.py` 增加反变换及单位标注。各种终点对应不同的效应定义，预测值应与终点和实验条件一起报告；log 值越高，对应达到该终点所需浓度越高，并不表示毒性效力越高。

## 关于 RF 和性能

附件仅含训练/测试数据及 `y1`，未含原始训练代码、RF 参数或已保存的模型。此程序使用 `RandomForestRegressor(n_estimators=300, max_features=1.0, min_samples_leaf=1, random_state=42)` 从 `traindata` 的 `y` 重新拟合。`y1` 对相同输入是一致的，符合原有预测值的表现，但不能单靠工作簿证明其生成方法；本程序的输出**不能宣称与原模型完全一致**。

在随附 `testdata` 上，新 RF 的 R² 约为 **0.868**，平均绝对误差约 **0.474 log 单位**。这不是独立验证结论：训练集中有 1,781 条完全重复记录，测试集中有 209 条完全重复记录，而且测试集的 1,080 行中有 621 行与训练集某行完全相同（涉及 416 种不同的重复记录）。该划分还可能共享化合物。正式发表或报告新化合物的泛化能力前，需要从原始记录按化合物/CAS 等合理分组重新划分、处理重复数据、在训练集内调参，然后在未参与训练的化合物上评估。现有 Excel 没有逐行 CAS，因此不能直接完成可靠的 CAS 分组评估。
