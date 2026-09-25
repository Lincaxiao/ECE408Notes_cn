# ECE 408 第 9 讲

## 编译与文件

从 `Lecture` 根目录运行：

```sh
latexmk Lecture9/ECE408_L09_2D_Tiled_Convolution_and_Reuse_Analysis_CN.tex
latexmk ECE408.tex
```

单讲与合订本读取同一份 `content.tex`，使用系列共用样式和讲次清单。需要 XeLaTeX、latexmk 和共用模板指定的字体。最终 PDF 与入口源码同级，辅助文件进入 `TexOutput/`。封面仅显示一次 ECE 408，以及讲次和英文主题。

原始压缩包、英文课件、全部图片与配套代码均保留。当前页数、结构、元数据、渲染检查和数值结果见本地 `REVIEW_LOG.md` 与 `verification.json`。历史记录位于 `sources/`，不代表当前成品的验收状态。

## 内容与来源

保留三种分块策略的讲解、策略 2 完整内核与启动接口、边界计数、复用推导和带宽例题。图片来源见 `asset_sources.md`。原包附带的根目录配置、样式和历史记录保存于 `sources/original_archive_root/`，不参与当前编译。

## 验证与实现条件

```sh
python3 Lecture9/code/verify_notes.py
```

脚本使用 Python 标准库，验证 8 组输入、102 个分块的索引与边界计数，以及复用、warp 分支和带宽例题。

CUDA 文件使用 C++17，启动文件是供已有程序调用的接口。输入和输出设备内存必须互不重叠；权重按行优先排列。调用方负责分配、同步、执行错误处理和结果传回。不同权重的并发调用还须协调对同一常量符号的访问。

本次环境无 `nvcc`，CUDA 编译、GPU 执行和测速均未验证；CPU 仿真不验证 GPU 并发、浮点舍入或实际访存事务。
