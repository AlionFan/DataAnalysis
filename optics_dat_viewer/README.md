# Optical DAT Viewer

用于光学 `.dat` 栅格数据的可视化与分析，支持：
- 文件上传与本地路径读取
- 2D 热力图、3D 表面图、X/Y 剖面分析
- ROI 区域选择、平滑与归一化
- 峰值检测、FWHM、积分面积
- 多文件归一化剖面对比
- CSV/PNG/JSON 导出

## 环境

```bash
conda activate DataAnalysis
pip install -r optics_dat_viewer/requirements.txt
```

## 启动

```bash
streamlit run optics_dat_viewer/app.py
```

## 示例文件

可在应用左侧输入本地文件路径，例如：
`/Users/fan/Downloads/guangyuan1000wan.dat`

## 说明

- 若导出 PNG 失败，请安装 `kaleido`。
- 处理模块在缺失 `scipy` 时会自动使用基础算法兜底。

## 作者与反馈

- 作者：胡一凡
- 邮箱：h1317483655@gmail.com
- 有问题请通过邮箱反馈。
