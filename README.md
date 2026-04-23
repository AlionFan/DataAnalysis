# Optical DAT Viewer

基于 Streamlit 的光学 `.dat` 栅格数据可视化与分析工具。

## 功能概览

### 数据加载
- 支持浏览器上传 `.dat` 文件（可多选）
- 支持输入本地文件路径加载
- 自动解析 DAT 文件头部（GRID 行数/列数、hole value、坐标增量与原点）并映射为二维矩阵

### 可视化模式（综合分析页面）
| 模式 | 说明 |
|------|------|
| **2D 热力图** | 带 Viridis 配色的热力图，可选对数色阶(log1p)、叠加等高线，支持交互标注工具 |
| **3D 表面图** | Plotly 3D Surface 渲染 |
| **剖面图分析** | X/Y 方向剖面提取，Savitzky-Golay 平滑、Min-Max 归一化，峰值检测(FWHM)、积分面积、高斯/多峰高斯拟合、1D FFT 频谱与低通重建 |
| **多文件对比** | 多文件剖面归一化叠加对比，可选按峰位对齐 |
| **时序对比** | 多帧热力图浏览与帧间差分图 |

### 频域分析
- 2D FFT 频谱幅度可视化
- 低通/高通/带通滤波与重建

### 数据统计
- **数据质量诊断**：NaN 比率、零值比率、饱和比率、异常值比率、均值、标准差
- **多 ROI 区域统计**：支持最多 6 个 ROI，分别统计 mean/std/sum/max
- **径向分箱统计**：以指定中心点做径向分箱，支持坐标偏移(offset)
- **半径均值计算**：以指定圆心和半径计算区域内均值/标准差/像素数/极值，支持欧式距离和曼哈顿距离两种模式
- **半径-强度曲线**：按曼哈顿距离（整数壳层）和欧几里得距离（精细分箱）绘制半径-均值强度曲线

### 数据编辑与修复
- 表格直接编辑：修改任意单元格数值，修改即时生效于后续分析
- 异常点交互修复：指定行列号，使用 3x3 邻域中值/均值替换

### AI 大模型分析
- 集成 SiliconFlow API，可选模型包括 MiniMax-M2.5、Kimi-K2.6、GLM-5.1、DeepSeek-V3.2
- 自动将质量诊断、ROI 统计、径向统计、峰值分析、批处理汇总作为上下文提交
- 支持上传补充分析文件（txt/csv/json）

### 批量处理
- 指定目录后自动扫描所有 `.dat` 文件，统计 mean/max/sum
- 实验批次对比看板（柱状图 + 折线图）
- 自动检测异常批次（|z| > 2）

### 流程模板
- 保存当前 ROI 范围、分箱参数等配置为 JSON 模板
- 加载已保存模板恢复参数

### 导出
- ROI 矩阵 CSV
- 峰值分析 CSV / JSON
- 当前图像 PNG（需 kaleido）
- 分析报告 HTML / Markdown

## 项目结构

```
optics_dat_viewer/
├── app.py            # Streamlit 主应用入口
├── dat_parser.py     # DAT 文件解析器（DatGrid 数据结构）
├── processing.py     # 核心算法：峰值检测、高斯拟合、FFT、质量诊断、径向统计等
├── plots.py          # Plotly 可视化：热力图、3D 表面图、等高线图、PNG 导出
├── ui.py             # Streamlit UI 组件：文件选择、表格编辑、剖面分析、多文件对比、ROI 统计
├── llm.py            # SiliconFlow 大模型调用封装
├── reports.py        # HTML/Markdown 报告生成、流程模板保存/加载、批量处理
├── requirements.txt  # Python 依赖
└── optics_dat_viewer/
    └── workflows/    # 流程模板 JSON 存储目录
```

## 安装与启动

### 1. 安装依赖

```bash
conda create -n DataAnalysis python==3.12
conda activate DataAnalysis
pip install -r optics_dat_viewer/requirements.txt
```

依赖清单：
- `streamlit` — Web 应用框架
- `numpy` / `pandas` — 数值计算与数据处理
- `scipy` — 峰值检测、曲线拟合、信号平滑（缺失时自动降级为基础算法）
- `plotly` — 交互式图表
- `kaleido` — PNG 导出（可选，缺失时 PNG 导出不可用）
- `openai` — SiliconFlow API 调用（AI 分析功能所需）
- `openpyxl` / `tabulate` — 表格格式支持
- `httpx[socks]` — 代理网络支持

### 2. 启动应用

```bash
streamlit run optics_dat_viewer/app.py
```

启动后浏览器自动打开 `http://localhost:8501`。

### 3. 加载数据

在左侧侧边栏中：
- 点击 **上传 .dat 文件** 选择一个或多个文件，或
- 在 **本地文件路径** 输入框中填写 `.dat` 文件的绝对路径

## 使用流程

1. **加载文件** — 侧边栏上传或输入路径
2. **选择 ROI** — 侧边栏 X/Y 范围滑块裁剪感兴趣区域
3. **选择可视化模式** — 顶部单选切换：热力图 / 3D 表面 / 剖面 / 多文件对比 / 时序
4. **分析与统计** — 查看质量诊断、ROI 统计、径向分箱、半径均值等
5. **修复异常** — 指定行列号，选择修复方式
6. **AI 分析**（可选）— 切换到 AI 大模型分析页面，输入 SiliconFlow API Key
7. **导出** — 底部导出 CSV / PNG / HTML / Markdown

## DAT 文件格式要求

工具解析的 `.dat` 文件格式如下：

```
GRID <rows> <cols>
<hole_value>
<row_delta> <col_delta>
<row_origin> <col_origin>
<data_matrix...>
```

- 第 1 行：`GRID` 关键字 + 行数 + 列数
- 第 2 行：hole value（等于此值的像素会被标记为 NaN）
- 第 3 行：行方向增量 + 列方向增量
- 第 4 行：行方向原点 + 列方向原点
- 第 5 行起：空格分隔的数值矩阵，维度须与头部 rows x cols 匹配
- 以 `!` 开头的行视为注释，自动跳过

## 注意事项

- 若 PNG 导出失败，请确认已安装 `kaleido`（`pip install kaleido`）
- 缺少 `scipy` 时，峰值检测和高斯拟合会使用内置基础算法兜底，精度可能降低
- AI 分析功能需要有效的 SiliconFlow API Key，可在 [SiliconFlow](https://cloud.siliconflow.cn/) 注册获取

## 作者与反馈

- 作者：胡一凡
- 邮箱：h1317483655@gmail.com
- 有问题请通过邮箱反馈
