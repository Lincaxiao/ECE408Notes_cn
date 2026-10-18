# 第14讲：GPU性能分析

正式文件：`ECE408_L14_Profiling_on_Nvidia_GPU_CN.tex`及同名PDF。单讲与合订本共用content.tex中的六节正文（10个分文件），沿用根目录共用样式与讲次元数据。

在Lecture根目录运行：

```sh
latexmk Lecture14/ECE408_L14_Profiling_on_Nvidia_GPU_CN.tex
python3 Lecture14/checks/verify_numbers.py
```

PDF与入口同级，辅助文件进入TexOutput。独立版30页（封面1页、目录1页、正文28页）。

英文课件位于sources，8张截图位于assets；原Lec14_source.zip保留在上级目录。原独立样式、字体、入口及旧说明归档于sources/original_package，不参与正式构建。来源及教材补充映射见SOURCE_NOTES.md，不编入成品。

数值脚本使用Python标准库，核对吞吐/峰值时间、资源预算、扇区/存储体计数、向量读取比例及新增尾部轮次例题，并检查原代码和截图完整性。没有重跑课件的GPU基准；没有执行CUDA编译、GPU运行、Sanitizer或Nsight采集。检查记录见REVIEW_LOG.md。
