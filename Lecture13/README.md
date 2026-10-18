# Lecture 13：原子操作与直方图

正式阅读文件为 `ECE408_L13_Atomic_Operations_and_Histogramming_CN.pdf`，同名 `.tex` 是单讲入口。正文由 `content.tex` 汇集，与合订本共用，格式及元数据从上一级目录加载。

在课程 Lecture 根目录执行：

```sh
latexmk Lecture13/ECE408_L13_Atomic_Operations_and_Histogramming_CN.tex
python Lecture13/checks/verify_histogram.py
```

最终 PDF 与入口同级；辅助文件位于 `Lecture13/TexOutput/`。原始压缩包 `../Lec13_source.zip` 保留全部便携字体、宏包和许可证。旧独立入口、样式及编译说明存于 `sources/original_archive/`，不参与当前构建。英文课件和原来源记录位于 `source/`。

正文保留基本全局原子与块级私有化两个完整 CUDA 核函数；教材补充包括四区间桶映射、线程内最近桶聚合、末段提交和访问顺序限制。聚合以伪代码说明，不作为第三套完整 CUDA 内核。

CPU 验证覆盖 420 个输入/网格组合、逐桶计数、唯一覆盖、初始化与合并覆盖、聚合末段、数值例题及四个代码块与实现文件的一致性。结果在 `checks/verification.json`。CUDA 编译、GPU 执行、竞争检测和测速未执行；CPU 检查不验证设备同步或性能。最新排版检查见 `REVIEW_LOG.md`。
