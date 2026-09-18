# ECE408 第 8 讲：Convolution Concept; Constant Cache

独立讲义 `ECE408_L08_Convolution_Concept_Constant_Cache_CN.pdf` 共 23 个 PDF 页面：封面 1 页、目录 1 页、正文 21 页。根目录 `ECE408.pdf` 包含第 1–8 讲，共 217 个 PDF 页面；第八讲从 PDF 第 197 页（正文页码 194）开始。

封面统一使用一处课程编号 `ECE 408`。合订本以大号课程编号作为唯一标题；单讲在课程编号下保留讲次和该讲英文标题。

## 源文件与资源

- 单讲入口仅加载共用样式、讲次元数据并调用第八讲正文；`content.tex` 是单讲与合订本共用的唯一正文顺序清单。
- `sections/` 保存六个正文部分；`assets/` 保存六张原图及 `sources.json` 来源记录。
- `ece408-lecture8-convolution-vk-FL26.pdf` 是原始 53 页英文课件。
- `sources/original_source_notes.tex` 保存原稿的来源与整合记录，仅用于追溯，不参与编译；其中页码和修订描述指原稿。
- `code/` 保存 CUDA 示例、CPU 索引仿真和最新仿真结果。
- `verification.json` 与 `REVIEW_LOG.md` 记录本次验收。

正文采用教材式中文，保留课程事项、定义、推导、数值例题、六张图和八个代码块。课件页码、出处标记与来源说明章节不编入 PDF。八个代码块的代码正文与原稿一致，使用不可分页容器；短标识符避免在行末拆开。

## 编译

需要 XeLaTeX、latexmk，以及系列共用的字体和宏包。字体沿用 STIX Two Text、STIX Two Math、Noto Sans Mono 和 Fandol，不使用系统专属字体路径。

在 `Lecture` 根目录运行：

```sh
latexmk Lecture8/ECE408_L08_Convolution_Concept_Constant_Cache_CN.tex
latexmk ECE408.tex
```

也可以运行 `Lecture8/build.sh` 编译单讲；脚本自动切换到系列根目录并使用现有 `latexmkrc`。最终 PDF 与入口 `.tex` 同级，辅助文件位于对应的 `TexOutput/`。

重新编译需要保留系列根目录中的 `style/`、`lectures.tex` 和 `latexmkrc`。单独复制 Lecture8 目录不构成完整构建环境。新增内容只修改相应正文文件，单讲与合订本读取同一份内容。

## 索引验证与 CUDA 示例

从系列根目录运行 CPU 索引仿真：

```sh
python3 Lecture8/code/validate_indexing.py
```

已通过 10,545 组比较，覆盖数值一致性、共享下标合法性、共享槽位恰好初始化一次、首尾边界、不满块、宽于输入的滤波器和非对称权重；主要一维、二维数值例题也已复算。

CUDA 示例包含直接版、常量版、策略 1 三段载入、策略 1 两轮载入和策略 3 核心区缓存，并提供主机初始化、错误处理和 CPU 参考解比较。策略 2 保留概念讲解，不新增完整实现。有 CUDA Toolkit 和适用 NVIDIA GPU 时可运行：

```sh
cd Lecture8/code
nvcc -std=c++17 -O2 convolution_examples.cu -o convolution_examples
./convolution_examples
```

本次环境没有 nvcc，未执行 CUDA 编译、GPU 运行或性能测试。讲义中的请求计数和逻辑字节量不代表实测 DRAM 流量或加速比。

## 文档验收

两个 PDF 均通过完整编译、qpdf 结构检查、US Letter 页面与元数据检查、书签目标及文字边界检查；无编译错误、Overfull、缺图、缺字或未解析引用。保留共用样式中 unicode-math 与 mathtools 的命令重叠提示，以及不影响阅读的 Underfull 提示，未将日志描述为完全无警告。

Lecture8 全部 23 页已渲染并逐页审阅；合订本全部 217 页已渲染并巡检缩略图，目录、章首页、长代码与复杂表格另作放大检查。最终修复了行内同步标识符跨页、同步说明段落分页和合订本公式汇总标题与表格分离的问题。
