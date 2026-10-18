# 第12讲：大语言模型中的计算

正式入口为 `ECE408_L12_Computation_in_Large_Language_Models_CN.tex`，PDF 同名并与入口同级。共用根目录模板和讲次元数据，`content.tex` 为单讲与合订本共用的五节正文清单。

从 Lecture 根目录编译：

```sh
latexmk Lecture12/ECE408_L12_Computation_in_Large_Language_Models_CN.tex
python3 Lecture12/checks/verify.py
```

辅助文件进入 `TexOutput/`。验证脚本需要 Python 3 和 NumPy，结果写入 `checks/results.json`。当前独立版25页（封面1页、目录1页、正文23页）。

内容涵盖词元与模型输入、因果多头注意力、CUDA 行归约、增量推理与逐层缓存、运算量与端到端性能。教材补充集中于变长缓存容量、批量的权重复用、完整数据传输的算术强度以及MHA/GQA/MQA的键值头数。

原英文课件、两张矢量图源码和CUDA代码保留。原包内独立模板、字体、texmf、构建脚本和旧检查结论保存在 `sources/original_package/`，不参与正式构建。原始压缩包仍在上级目录。来源及技术口径记录见 `sources.json`，不编入成品。

CPU验证覆盖线程分工、稳定因果softmax、掩码非有限值、非整块长度、缓存一致性、在线分块归一化及数值例题。当前没有nvcc，未执行CUDA编译、GPU运行或测速。完整记录见 `REVIEW_LOG.md`。
