# ECE 408 中文讲义

ECE 408 课程的中文讲义与 LaTeX 源码，目前包含第 1–8 讲。

[阅读完整合订本 ECE408.pdf](ECE408.pdf)

| 讲次 | 单讲 PDF |
| :-- | :-- |
| 1 | [Parallel Programming Introduction](Lecture1/ECE408_L01_Parallel_Programming_Introduction_CN.pdf) |
| 2 | [Introduction to CUDA C and Data Parallel Programming](Lecture2/ECE408_L02_Introduction_to_CUDA_C_and_Data_Parallel_Programming_CN.pdf) |
| 3 | [CUDA Parallel Execution Model Multidimensional Grids and Data](Lecture3/ECE408_L03_CUDA_Parallel_Execution_Model_Multidimensional_Grids_and_Data_CN.pdf) |
| 4 | [Compute Architecture and Scheduling](Lecture4/ECE408_L04_Compute_Architecture_and_Scheduling_CN.pdf) |
| 5 | [CUDA Memory Model](Lecture5/ECE408_L05_CUDA_Memory_Model_CN.pdf) |
| 6 | [Data Locality and Tiled Matrix Multiply](Lecture6/ECE408_L06_Data_Locality_and_Tiled_Matrix_Multiply_CN.pdf) |
| 7 | [DRAM Bandwidth and Other Performance Considerations](Lecture7/ECE408_L07_DRAM_Bandwidth_and_Other_Performance_Considerations_CN.pdf) |
| 8 | [Convolution Concept Constant Cache](Lecture8/ECE408_L08_Convolution_Concept_Constant_Cache_CN.pdf) |

## 编译

需要 XeLaTeX、latexmk，以及 STIX Two Text、STIX Two Math、Noto Sans Mono 和 Fandol 字体。所有命令从仓库根目录运行：

```sh
latexmk ECE408.tex
latexmk Lecture8/ECE408_L08_Convolution_Concept_Constant_Cache_CN.tex
```

共用样式位于 `style/`，讲次清单位于 `lectures.tex`；各讲 `content.tex` 引入 `sections/` 中的正文，单讲与合订本使用同一份内容。最终 PDF 与主 `.tex` 同级，编译辅助文件进入 `TexOutput/`。新增和修订讲义遵循 [讲义模板](LECTURE_NOTES_TEMPLATE.md)。

## 跟踪范围

Git 跟踪讲义源码、共用样式、构建配置、必要图片及来源索引、示例代码、维护文档，以及中文单讲和合订本 PDF。PDF 是直接阅读的交付文件，因此与源码一同维护。

完整原版课件不纳入仓库，也不随讲义分发；原文件仅保留在本地。

原始英文课件、旧草稿、编辑器设置、编译缓存、临时文件、压缩包、机器相关验收日志和自动生成的仿真结果留在本地，不纳入版本控制。部分单讲说明中提到的原始材料与验收记录属于这些本地文件；重新编译讲义不依赖它们。

## 示例验证

无需 GPU 的一维卷积索引检查：

```sh
python3 Lecture8/code/validate_indexing.py
```

CUDA 示例运行需要另行配置 CUDA Toolkit 和兼容的 NVIDIA GPU。
