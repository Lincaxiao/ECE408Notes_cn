# ECE 408 第 10 讲

## 编译与文件

从 `Lecture` 根目录运行：

```sh
latexmk Lecture10/ECE408_L10_Introduction_to_ML_Inference_and_Training_in_DNNs_CN.tex
latexmk ECE408.tex
```

单讲与合订本读取同一份 `content.tex`，使用系列共用样式和讲次清单。需要 XeLaTeX、latexmk 和共用模板指定的字体。最终 PDF 与入口源码同级，辅助文件进入 `TexOutput/`。封面仅显示一次 ECE 408，以及讲次和英文主题。

原始压缩包、英文课件、全部图片与配套代码均保留。当前页数、结构、元数据、渲染检查和数值结果见本地 `REVIEW_LOG.md` 与 `verification.json`。历史记录位于 `sources/`，不代表当前成品的验收状态。

## 内容与来源

保留表示学习、感知机、全连接层、激活函数、softmax、反向传播、小批量和泛化的完整讲解。两层例题使用双层 sigmoid 后接 softmax，以及概率加权数字距离平方损失。

来源附录保存在 `sources/original_source_notes.tex`；必要数学条件已融入正文，附录不编入成品。图片来源见 `assets/manifest.json`。

## 数值验证

```sh
python3 Lecture10/checks/verify_gradients.py
```

需要 NumPy。脚本检查全部 15 个参数的中心差分梯度、一次更新后的损失下降、四种 XOR 输入与 7,960 个参数的计数，写入 `checks/numeric_results.json`。该小网络是公式核算例题，不是 MNIST 训练结果。本次未进行 GPU 训练或性能测试。
