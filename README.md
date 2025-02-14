## Awesome-CIL

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" style="zoom: 60%;" /></div>
<p></p>
<div align=center><img src="https://visitor-badge.laobi.icu/badge?page_id=Geeks-Z.Awesome-CIL&left_color=green&right_color=red" /> <img src="https://img.shields.io/github/last-commit/Geeks-Z/Awesome-CIL" /> <img src="https://img.shields.io/github/license/Geeks-Z/Awesome-CIL" /></div>

## 🎉 Introduction

- **增量学习**：Continual Learning/Incremental Learning/Life-Long Learning

- **汇总**类增量学习（CIL，Class-Incremental Learning）的论文和代码，并对论文进行复现

- **论文阅读笔记**：[论文阅读](https://www.zhihu.com/column/c_1847944474761760768)

## 🚀 Survey

| Title                                                        | Venue | Year | Code                                                   |
| ------------------------------------------------------------ | ----- | ---- | ------------------------------------------------------ |
| [Class-Incremental Learning: A Survey](http://arxiv.org/abs/2302.03648) | TPAMI | 2024 | [Official](https://github.com/zhoudw-zdw/CIL_Surve)    |
| [Continual Learning with Pre-Trained Models: A Survey](http://arxiv.org/abs/2401.16386) | IJCAI | 2024 | [Official](https://github.com/sun-hailong/LAMDA-PILOT) |
| [PyCIL: A Python Toolbox for Class-Incremental Learning](https://arxiv.org/abs/2112.12533) |       |      | [Official](https://github.com/G-U-N/PyCIL)              |

## 🌟 Papers

>分类准则基于 [Class-Incremental Learning: A Survey](https://arxiv.org/abs/2302.03648)，稍有改动！

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20241017175011.png" /></div>

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20241017175741.png" /></div>

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20241017175130.png" /></div>

| Title                                                        | Venue | Year | Type               | Code                                                    |
| ------------------------------------------------------------ | ----- | ---- | ------------------ | ------------------------------------------------------- |
| [MOS: Model Surgery for Pre-Trained Model-Based Class-Incremental Learning](http://arxiv.org/abs/2412.09441) | AAAI  | 2025 | PEFT Expansion     | [Official](https://github.com/sun-hailong/AAAI25-MOS)   |
| [Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning](http://arxiv.org/abs/2403.12030) | CVPR  | 2024 | PEFT Expansion     | [Official](https://github.com/sun-hailong/CVPR24-Ease)  |
| [Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need](https://arxiv.org/pdf/2303.07338) | IJCV  | 2024 | PEFT Expansion     | [Official](https://github.com/zhoudw-zdw/RevisitingCIL) |
| [CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning](http://arxiv.org/abs/2211.13218) | CVPR  | 2023 | PEFT Expansion     | [Official](https://github.com/GT-RIPL/CODA-Prompt)      |
| [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/abs/2204.04799) | ECCV  | 2022 | PEFT Expansion     | [Official](https://github.com/google-research/l2p)      |
| [Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654) | CVPR  | 2022 | PEFT Expansion     | [Official](https://github.com/google-research/l2p)      |
| [FOSTER: Feature Boosting and Compression for Class-Incremental Learning](https://arxiv.org/abs/2204.04662) | ECCV  | 2022 | Backbone Expansion | [Official](https://github.com/G-U-N/ECCV22-FOSTER)      |
| [DER: Dynamically Expandable Representation for Class Incremental Learning](2021) | CVPR  | 2021 | Backbone Expansion | [Official](https://github.com/G-U-N/ECCV22-FOSTER)      |
| [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725) | CVPR  | 2017 | Memory             | [Official](https://github.com/srebuffi/iCaRL)           |

## 📚 Datasets

| Dataset       | training instances | testing instances | Classes | Link                                                         |
| ------------- | ------------------ | ----------------- | ------- | ------------------------------------------------------------ |
| CIFAR100      | 50,000             | 10,000            | 100     |                                                              |
| CUB200        | 9,430              | 2,358             | 200     |                                                              |
| ImageNet-R    | 24,000             | 6,000             | 200     |                                                              |
| ImageNet-A    | 5,981              | 1,519             | 200     |                                                              |
| ObjectNet     | 26,509             | 6,628             | 200     | [https://objectnet.dev/download.html](https://objectnet.dev/download.html) |
| Omnibenchmark | 89,697             | 5,983             | 300     |                                                              |
| VTAB          | 1,796              | 8,619             | 50      |                                                              |

## 📊 Reproduced Results

> 实验结果：三次实验的平均准确率（Accuracy）和标准差（Std）/ 论文中的结果

### Code

> [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20241226204218.png" style="zoom: 80%;" /></div>

### Dataset split

- `B-$m$ Inc-$n$' ：$m$代表初始增量阶段类别数量，$n$ 代表增量阶段；如果$m=0$，则代表所有类别均分为$n$个增量阶段，如果$m\neq0$，则初始阶段为$m$个类别，其他类别均分为$n$个增量阶段；
- LFH(learning from half)，表示在模型训练的初始阶段先用一半的类别进行训练，然后剩下一半的类别均匀分为N个阶段进行训练；
- LFS(learning from scratch)，表示所有的类别均匀地分为N个阶段进行训练

### Backbone

`ViT-B/16-IN21K`

### Memory

For exemplar parameters, DER, iCaRL and FOSTER set the `fixed_memory` option to false and retain the `memory_size` of 2000 for CIFAR100, while setting `fixed_memory` option to true and retaining the `memory_per_class` of 20 for ImageNet-R. On the contrary, other models are exemplar-free.

### 🗂️ Dependencies

- pytorch 2.0.1
- torchvision 0.15.2
- timm 0.6.12
- tqdm  4.65.0
- numpy 1.21.5
- scipy 1.10.1
- easydict 1.13

### CIFAR-100

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20250104174921.png" style="zoom: 80%;" /></div>

|             | B0 Inc5           | B0 Inc10    | B0 Inc20    | B50 Inc5    | B50 Inc10   |
| ----------- |-------------------| ----------- | ----------- | ----------- | ----------- |
|MOS | 94.81 ± 0.02/93.3 |94.71 ± 0.05|93.37 ± 0.07|94.09 ± 0.0|94.16 ± 0.01|
|Ease | 93.1 ± 0.01/91.51 | 92.55 ± 0.01| 91.64 ± 0.02| 90.39 ± 0.03| 89.25 ± 0.04|
|CODA-Prompt | 92.85 ± 0.0       |91.56 ± 0.05|88.87 ± 0.02|86.09 ± 0.2|77.74 ± 0.05|
|DualPrompt | 91.39 ± 0.01      |90.28 ± 0.04|88.52 ± 0.01|87.49 ± 0.0|81.02 ± 0.02|
|L2P | 90.72 ± 0.06      |89.85 ± 0.01|87.58 ± 0.0|87.53 ± 0.06|79.43 ± 0.01|
|SimpleCIL | 81.12 ± 0.0       |82.31 ± 0.0|82.79 ± 0.0|78.66 ± 0.0|78.54 ± 0.0|
|Finetune | 81.48 ± 0.0       |76.93 ± 0.0|72.14 ± 0.0|82.2 ± 0.0|79.99 ± 0.0|
|FOSTER-CNN | 93.74 ± 0.0       |93.68 ± 0.0|93.71 ± 0.0|92.2 ± 0.0|92.15 ± 0.0|
|FOSTER-NME | 93.7 ± 0.0        |93.9 ± 0.0|93.69 ± 0.0|92.26 ± 0.0|92.22 ± 0.0|
|DER-CNN | 88.78 ± 0.0       |88.58 ± 0.0|88.63 ± 0.08|86.92 ± 0.13|86.55 ± 0.01|
|DER-NME | 91.36 ± 0.0       |91.01 ± 0.0|90.96 ± 0.15|89.41 ± 0.04|89.01 ± 0.05|
|iCaRL-CNN | 87.14 ± 0.14      |85.91 ± 0.12|84.43 ± 0.25|83.4 ± 0.04|81.32 ± 0.46|
|iCaRL-NME | 90.8 ± 0.07       |90.29 ± 0.05|89.46 ± 0.02|88.92 ± 0.06|87.39 ± 0.23|

### CUB-200

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20250104175008.png" style="zoom: 80%;" /></div>

|             | B0 Inc5     | B0 Inc10           | B0 Inc20    | B100 Inc5   | B100 Inc10  |
| ----------- | ----------- |--------------------| ----------- | ----------- | ----------- |
|MOS |92.75 ± 0.0| 93.27 ± 0.0/93.49  |93.48 ± 0.0|91.45 ± 0.0|91.43 ± 0.0|
|Ease |90.14 ± 0.05| 90.21 ± 0.01/92.23 | 91.32 ± 0.01| 85.26 ± 0.02| 87.44 ± 0.07|
|CODA-Prompt |85.78 ± 0.09| 84.93 ± 0.05       |83.03 ± 0.08|79.59 ± 0.03|76.44 ± 0.14|
|DualPrompt |85.5 ± 0.03| 84.84 ± 0.04       |82.9 ± 0.08|77.01 ± 0.07|73.8 ± 0.1|
|L2P |83.74 ± 0.04| 81.8 ± 0.05        |79.33 ± 0.01|79.15 ± 0.18|74.65 ± 0.02|
|SimpleCIL |89.92 ± 0.0| 90.57 ± 0.01       |90.96 ± 0.0|87.51 ± 0.04|87.43 ± 0.04|
|Finetune |77.36 ± 0.49| 70.94 ± 0.74       |60.21 ± 2.64|72.88 ± 0.42|62.32 ± 2.26|
|FOSTER-CNN |78.81 ± 0.53| 81.18 ± 0.36       |85.17 ± 0.64|82.65 ± 0.25|85.56 ± 0.01|
|FOSTER-NME |90.65 ± 0.05| 91.02 ± 0.16       |91.4 ± 0.34|88.92 ± 0.18|88.72 ± 0.02|
|DER-CNN |89.08 ± 0.48| 88.71 ± 0.86       |90.13 ± 0.05|87.14 ± 0.2|86.72 ± 0.06|
|DER-NME |89.89 ± 0.19| 89.43 ± 0.71       |90.99 ± 0.05|88.62 ± 0.11|88.34 ± 0.21|
|iCaRL-CNN |87.77 ± 0.25| 88.2 ± 0.13        |88.2 ± 0.36|85.93 ± 0.32|85.4 ± 0.42|
|iCaRL-NME |89.06 ± 0.04| 89.94 ± 0.13       |90.2 ± 0.21|87.91 ± 0.14|87.25 ± 0.28|

### ImageNet-R

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20250104175051.png" style="zoom: 80%;" /></div>

|             | B0 Inc5 | B0 Inc10 | B0 Inc20 | B100 Inc5 | B100 Inc10 |
| ----------- | ------- | -------- | -------- | --------- | ---------- |
|MOS |82.79 ± 0.0|82.63 ± 0.0|81.28 ± 0.0|81.09 ± 0.0|81.11 ± 0.0|
|Ease |81.55 ± 0.04| 81.01 ± 0.07| 78.88 ± 0.01| 77.96 ± 0.01| 76.58 ± 0.01|
|CODA-Prompt |79.9 ± 0.06|78.72 ± 0.06|74.45 ± 0.13|73.1 ± 0.03|67.86 ± 0.02|
|DualPrompt |73.54 ± 0.02|71.16 ± 0.13|69.84 ± 0.06|65.04 ± 0.16|59.73 ± 0.13|
|L2P |76.88 ± 0.02|76.31 ± 0.04|73.69 ± 0.01|69.77 ± 0.05|64.88 ± 0.0|
|SimpleCIL |65.86 ± 0.02|67.07 ± 0.01|67.59 ± 0.01|63.53 ± 0.01|63.41 ± 0.01|
|Finetune |72.24 ± 0.42|68.32 ± 0.46|61.95 ± 1.74|70.73 ± 0.02|66.72 ± 0.28|

### ImageNet-A

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20250104175133.png" style="zoom: 80%;" /></div>

|             | B0 Inc5 | B0 Inc10 | B0 Inc20 | B100 Inc5 | B100 Inc10 |
| ----------- | ------- | -------- | -------- | --------- | ---------- |
|MOS |65.56 ± 0.0|66.8 ± 0.0|64.14 ± 0.0|64.93 ± 0.0|64.88 ± 0.0|
|Ease |67.84 ± 0.08| 62.67 ± 0.06| 58.13 ± 0.32| 63.55 ± 0.35| 62.18 ± 0.0|
|CODA-Prompt |60.99 ± 0.01|56.31 ± 0.17|47.62 ± 0.11|56.24 ± 0.26|53.2 ± 0.2|
|DualPrompt |54.16 ± 0.1|52.43 ± 0.03|48.59 ± 0.1|47.02 ± 0.11|45.05 ± 0.33|
|L2P |53.41 ± 0.12|52.68 ± 0.08|45.93 ± 0.18|49.32 ± 0.14|46.85 ± 0.08|
|SimpleCIL |58.38 ± 0.21|59.56 ± 0.16|60.35 ± 0.21|53.51 ± 0.16|53.43 ± 0.16|
|Finetune |38.03 ± 0.35|33.73 ± 1.2|15.26 ± 5.64|30.32 ± 0.59|24.65 ± 0.08|

### Omnibenchmark

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20250104175214.png" style="zoom: 80%;" /></div>

|             | B0 Inc5     | B0 Inc10    | B0 Inc20    | B0 Inc30    | B150 Inc5   | B150 Inc10  |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|MOS |85.59 ± 0.0|85.93 ± 0.0|85.93 ± 0.0|86.04 ± 0.0|83.51 ± 0.0|83.26 ± 0.0|
|Ease |75.79 ± 0.0| 74.84 ± 0.0| 74.75 ± 0.0| 74.24 ± 0.0| 71.68 ± 0.0| 69.18 ± 0.0|
|CODA-Prompt |80.04 ± 0.0|77.01 ± 0.0|75.14 ± 0.0|74.31 ± 0.0|70.36 ± 0.0|63.14 ± 0.0|
|DualPrompt |79.37 ± 0.01|74.93 ± 0.05|74.84 ± 0.02|73.58 ± 0.06|71.51 ± 0.0|63.95 ± 0.0|
|L2P |77.99 ± 0.0|74.15 ± 0.02|72.72 ± 0.08|71.13 ± 0.04|70.17 ± 0.01|63.38 ± 0.02|
|SimpleCIL |77.82 ± 0.07|78.58 ± 0.05|79.24 ± 0.04|79.52 ± 0.04|74.53 ± 0.04|74.39 ± 0.05|
|Finetune |70.43 ± 0.16|65.85 ± 0.51|60.82 ± 0.03|57.61 ± 0.29|68.13 ± 0.17|64.59 ± 0.34|
|FOSTER-CNN |77.97 ± 0.14|78.08 ± 0.28|81.16 ± 0.15|81.53 ± 0.24|79.1 ± 0.0|79.91 ± 0.08|
|FOSTER-NME |83.71 ± 0.21|82.81 ± 0.27|83.31 ± 0.12|82.91 ± 0.07|81.52 ± 0.14|81.2 ± 0.04|
|DER-CNN |78.17 ± 0.48|77.64 ± 0.08|77.11 ± 0.33|77.16 ± 0.0|77.03 ± 0.16|76.46 ± 0.04|
|DER-NME |81.72 ± 0.4|80.78 ± 0.03|80.09 ± 0.18|79.24 ± 0.0|79.55 ± 0.21|78.65 ± 0.13|
|iCaRL-CNN |76.43 ± 0.26|74.73 ± 0.38|73.81 ± 0.3|73.15 ± 0.56|74.11 ± 0.2|72.38 ± 0.11|
|iCaRL-NME |82.17 ± 0.24|80.93 ± 0.01|80.01 ± 0.16|79.56 ± 0.15|79.12 ± 0.01|78.06 ± 0.25|

### VTAB

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20250104175243.png" style="zoom: 80%;" /></div>

|             | B0 Inc5 | B0 Inc10          |
| ----------- | ------- |-------------------|
|MOS |92.59 ± 0.0| 93.18 ± 0.0/92.62 |
|Ease |90.47 ± 0.0| 91.07 ± 0.0/93.61      |
|CODA-Prompt |87.82 ± 0.0| 83.36 ± 0.0       |
|DualPrompt |89.96 ± 0.0| 87.61 ± 0.0       |
|L2P |82.11 ± 0.0| 78.04 ± 0.0       |
|SimpleCIL |90.94 ± 0.0| 91.66 ± 0.0       |
|Finetune |78.69 ± 0.0| 59.2 ± 0.0        |

> MOS Ease Inc10结果比Inc5更好？

## 👨‍🏫 TODO

| Title                                                                                                                                                   | Venue | Year | Type            | Code                                                        |
|---------------------------------------------------------------------------------------------------------------------------------------------------------| ----- | ---- | --------------- | ----------------------------------------------------------- |
| [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](http://arxiv.org/abs/2404.00228)                                                | CVPR  | 2024 | PTM             | [Official](https://github.com/liangyanshuo/InfLoRA)         |
| [FCS: Feature Calibration and Separation for Non-Exemplar Class Incremental Learning](https://ieeexplore.ieee.org/document/10657158/?arnumber=10657158) | CVPR  | 2024 | Feature Rectify | [Official](https://github.com/zhoujiahuan1991/CVPR2024-FCS) |
|                                                                                                                                                     |       |      |                 |                                                             |
|                                                                                                                                                         |       |      |                 |                                                             |

### Different PTMs

| PTM             | Pre-Trained Dataset | Finetuned Dataset |
| --------------- | ------------------- | ----------------- |
| ViT-B/16-IN1K   | ImageNet21K         | ImageNet1K        |
| ViT-B/16-IN21K  | ImageNet21K         | -                 |
| ViT-L/16-IN1K   | ImageNet21K         | ImageNet1K        |
| ViT-B/16-DINO   | ImageNet            | -                 |
| ViT-B/16-SAM    | SA-1B (Segment Anything 1 Billion Dataset) | - |
| ViT-B/16-MAE    | ImageNet21K         | -                 |
| ViT-B/16-CLIP   | OpenAI CLIP Dataset (a large corpus of text-image pairs) | - |
| ResNet18/50/152 | ImageNet1K          | -                 |

### Parameters

> 20Epoch batch_size=48 memory_size: 2000

| Method      | Tunable Parameters（Backbone） | All Parameters | Average Accuracy (%)<br />(CIFAR B0 Inc5) |
| ----------- | ------------------------------ | -------------- | ----------------------------------------- |
| Ease        |                                |                |                                           |
| SimpleCIL   |                                |                |                                           |
| CODA-Prompt |                                |                |                                           |
| DualPrompt  |                                |                |                                           |
| L2P         |                                |                |                                           |
| FOSTER      |                                |                |                                           |
| DER         |                                |                |                                           |
| iCaRL       |                                |                |                                           |
| Finetune    |                                |                |                                           |

## 🤗 Acknowledgments

- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
- [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
