# Lec14 来源与核验记录

本文件独立保存，不编入 正式PDF。

## 原始材料

- 用户提供的课件：`sources/ece408-lecture14-profiling-FL26.pdf`，77 页，标题为 **Profiling on Nvidia GPU**。
- 用户提供的格式要求：`sources/LECTURE_NOTES_TEMPLATE.md`。
- 表格中三组设备的测量时间、吞吐率、加速比、效率与报告计数来自原课件，不是本次运行实验的结果。

## 正文截图

下表页码均为原始 PDF 从 1 开始的物理页码。图片只裁切原图，没有更改其中数值。屋顶线示意图 为根据正文定义直接使用 TikZ 绘制的屋顶线示意图，不是实测结果。

| 正文图号 | 文件 | 原 PDF 页码 | 内容 |
| --- | --- | ---: | --- |
| 1 | `assets/l1_path.png` | 15 | L1／纹理单元与共享路径 |
| 2 | `assets/nsys_timeline.png` | 36 | 主机、CUDA API 与 GPU 时间线 |
| 4 | `assets/memory_chart.png` | 47 | 简单内核的存储工作负载图 |
| 5 | `assets/scheduler.png` | 50 | 每调度器的线程束统计 |
| 6 | `assets/roofline_comparison.png` | 61 | 多种矩阵乘法实现的屋顶线比较 |
| 7 | `assets/vector_sass_detail.png` | 76 | LDS.128 与标量 LDS 指令片段 |
| 8 | `assets/instruction_comparison.png` | 74 | 共享改进版与原版的指令组合 |
| 9 | `assets/stall_comparison.png` | 73 | 改进前后的线程束状态比较 |

## 数值与源材料中的技术问题

| 原材料位置 | 原材料内容或问题 | 讲义中的处理及依据 |
| --- | --- | --- |
| 第 25–29 页 | 三组设备的基准结果及不同精度的舍入值 | 保留实测列。显示时间、吞吐率与加速比独立舍入，正文说明由显示值重算可能有偏差。 |
| 第 28 页 | L40S 的理论时间约 0.05 ms、理论加速比约 69.4 倍，与表中峰值及共同工作量不一致 | 使用约 6.72×10^9 FLOP 和 65,434 GFLOP/s 重新计算，得到约 0.1027 ms，以及相对于 3.47 ms 的约 33.8 倍理想加速比。此处为算术校验；未用其他型号的外部峰值替换原材料分母。 |
| 第 12 页等 | 共享内存和 L1、寄存器和线程束的简化硬件表述容易混淆 | 区分 CUDA 存储空间、物理资源位置、共享容量和程序可见语义。寄存器预算以 SM 的资源分配条件解释。外部依据见下列 CUDA 指南。 |
| 第 42 页 | `cudaProfilerEnd()` | 采用官方接口 `cudaProfilerStop()`，并解释 `--profile-from-start off`。控制接口作用于当前上下文，例子明确没有其他流并发提交的前提。 |
| 第 56 页 | 前文列出 32 个共享存储体，随后一处写成 16 | 单精度基本模型统一为 32 个存储体，并显式给出地址映射条件。 |
| 第 70 页 | 与向量读取相关的注释将宽度写成 128 bytes | 区分每线程 128 bits = 16 bytes，与常见共享服务阶段的 128 bytes。对 32 个线程的 512 bytes 宽请求，说明四个理想波前不自动表示冲突。 |
| 第 50、53、68 页 | 每调度器 12 个线程束、每 SM 48 个线程束等设备相关数值 | 保留为特定报告示例，避免写成所有 GPU 的统一规格。25.18 为线程束周期／发射口径，包含被选中及其他状态，不能全称为等待。 |
| 第 60–62 页 | 高 SoL 汇总利用率、全局请求下降与 DRAM 读取上升 | 区分资源汇总、浮点峰值效率、逻辑访问与分层物理流量；不从汇总指标直接断言 A/B 转置错误。 |
| 第 70–76 页 | 共享读取向量化与报告变化 | 给出类型明确的 float4 消费端示例，明确加载、边界补零和同步的外层条件；推导 8→5 次共享读取及 37.5% 降幅，保留由 SASS 确认最终生成宽指令的要求。 |

关于矩阵大小：仅凭表格及启动配置，不能唯一确定 M、N、K；正文只回推约 6.72×10^9 FLOP 的共同工作量，没有补造未给出的矩阵维度。

## 外部技术核验

核验日期：2026-10-17。优先使用 NVIDIA 官方文档，范围限于接口名称、工具选项、统计口径与硬件机制。工具版本之间可能有差别，正文保留了版本或硬件条件，不把当前文档的所有特性反推为原实验环境的行为。

| 官方资料 | 核验用途 |
| --- | --- |
| [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | SoL 汇总、存储工作负载、扇区命中率、线程束调度、停顿、重放及测量扰动 |
| [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) | 内核过滤、launch-skip / launch-count、NVTX 过滤需同时启用 `--nvtx`、采集范围 |
| [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | 系统时间线、采集命令与交互会话；旧版本 `cancel` 选项保留条件表述 |
| [CUDA Runtime Profiler Control](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__PROFILER.html) | `cudaProfilerStart()`、`cudaProfilerStop()` 及上下文范围 |
| [NVCC Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) | `-lineinfo`、`-g`、`-G` 的不同作用，以及性能分析构建中设备调试选项的影响 |
| [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | 正确性、事件计时、共享存储体、合并访存与占用率条件 |
| [CUDA Programming Guide: Writing CUDA Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) | 线程块、流、地址空间与 CUDA 内核语义 |
| [NVIDIA: Roofline Analysis with Nsight Compute](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/) | 实测算术强度、分层带宽上界与计算上界 |
| [NVIDIA Developer Forums: bank conflict test](https://forums.developer.nvidia.com/t/problem-about-bank-conflict-test/285476) | NVIDIA 员工对 LDS.128 理想波前与额外波前的说明，作为官方指南的细节补充 |

## 补充推导与代码边界

- 屋顶线的时间下界、矩阵乘法的理想流量、扇区数例题、存储体冲突的最大公约数推导，以及资源预算例题，为帮助理解而补全；不是额外课程规定。
- 资源预算例子是明确条件下的简化模型。每块 16 KiB 指预算中的共享内存总占用量，已与用户声明大小区分；真实设备还需考虑分配取整和保留资源。
- CUDA 事件计时例子包含错误检查、预热、重复执行和同流启动示例；向量读取代码明确是消费阶段的辅助函数，不省略必要条件后声称是完整的改进内核。
- 未将现代工具版本的测量结果与课件中的基准表混写，也未宣称在本次环境中实际运行了 GPU 基准或 Sanitizer。

## 本次系列整合（2026-10-18）

- 共用模板接入，仅保留6个实际章节，10份正文文件的分工不变。原独立模板、字体、入口与旧说明归档于sources/original_package，不参与构建。
- 教材Ch06《Performance considerations》印刷第143页的线程粗化尾部轮次，补入第5节：1584块、每轮264块原有6满轮，粗化后396块形成1满轮加半轮，等时理想槽位利用率75%。保持数据规模、设备资源和每块耗时的条件说明。
- 原有向量读取、资源预算、访存合并和性能策略已覆盖教材对应要点，不额外增加完整高级矩阵乘法实现。
- 说明报告数据舍入后可能无法重现相对变化（1.24/1.23GB与-1.34%）。原实测列保留，不作为当前重跑。
- 本次未新增或修改CUDA/命令代码，12段listing正文与原包完全相同；8张截图字节完全相同。仅检查保存完整性和数值推导，无新增原代码正确性测试。
- 当前验证见REVIEW_LOG.md与checks/numerical_results.json；旧来源文件留作历史记录。
