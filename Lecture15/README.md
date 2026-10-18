# Lecture 15：并行计算模式与归约树

正式 PDF 和入口同名：`ECE408_L15_Parallel_Computation_Patterns_Reduction_Trees_CN`。单讲和合订本使用同一 `content.tex` 与五节正文，格式及讲次元数据从上一级加载。

从课程 Lecture 根目录执行：

```sh
latexmk Lecture15/ECE408_L15_Parallel_Computation_Patterns_Reduction_Trees_CN.tex
python Lecture15/checks/verify_examples.py
```

PDF 与入口同级，辅助文件在 `TexOutput/`。原始 `../Lec15_source.zip` 保留全部字体及许可证；旧入口、样式、构建脚本与旧说明归档于 `sources/original_archive/`，不参与当前构建。英文课件、全部四个外部图形资产及 `examples/reduction_examples.cuh` 均保留。

本次只补充整数中间溢出条件与算术通道利用率，保留现有实现深度。11 个 CUDA 代码块的内容及配套实现文件保持不变，未重复执行其正确性验证；脚本检查代码保留情况、正文索引边界、计数和数值例题。CUDA 编译、GPU 执行和测速未运行。详见 `REVIEW_LOG.md` 和 `checks/verification.json`。
