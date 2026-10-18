# Lecture 11：卷积神经网络的计算

英文题名为 *Computation in Convolutional Neural Networks*，原课件日期为 2026 年 9 月 29 日。正式入口为 `ECE408_L11_Computation_in_Convolutional_Neural_Networks_CN.tex`，同名 PDF 位于本目录。

正文由 `content.tex` 唯一列出，包含 5 个技术章节。单讲和合订本共用正文及根目录 `style/ece408-common.tex`，讲次元数据来自根目录 `lectures.tex`。不启用原压缩包附带的独立样式与字体。原始压缩包保留于上级目录，原英文课件及编辑记录保留于 `source/`。

在 Lecture 根目录编译：

```sh
latexmk Lecture11/ECE408_L11_Computation_in_Convolutional_Neural_Networks_CN.tex
```

沿用根目录 `latexmkrc`，最终 PDF 与入口同级，辅助文件进入本目录 `TexOutput/`。

正文保留 CNN 结构、卷积与池化、直接 CUDA 映射、输入展开、隐式展开、项目规模与全部例题。补充步幅与零填充的尺寸推导，以及矩阵三个维度对分块利用率的约束。后续代码仍采用步幅 1、无填充、无空洞的方形核；未将一般尺寸公式误作现有代码支持范围。

在 Lecture 根目录重跑验证：

```sh
python3 Lecture11/checks/verify.py
```

驱动直接编译并执行 `code/` 的 CPU 函数，启用 C++17 严格警告和 UndefinedBehaviorSanitizer。覆盖卷积、池化、展开与 GEMM 一致性、线程覆盖、批次布局、大偏移、正文算例和新增尺寸公式。结果写入 `checks/verification.json`。AddressSanitizer 在当前 macOS 的运行时初始化阶段阻塞，未完成执行；记录中明确区分该限制。原包旧结果单独保存在 `source/original_checks/`，不作为本次验证结论。

CUDA 示例保持原程序语义；CPU 枚举检验不能替代 CUDA 编译或 GPU 执行。本次未开展 CUDA 编译、GPU 执行或性能测试。文档编译、页面数及视觉核查记录见 `REVIEW_LOG.md`。
