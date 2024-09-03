## Awesome-CIL

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" style="zoom: 60%;" /></div>
<p></p>
<div align=center><img src="https://visitor-badge.laobi.icu/badge?page_id=Geeks-Z.Class-Incremental-Learning&left_color=green&right_color=red" /> <img src="https://img.shields.io/github/last-commit/Geeks-Z/Class-Incremental-Learning" /> <img src="https://img.shields.io/github/license/Geeks-Z/Class-Incremental-Learning" /></div>

## 🎉 介绍

CIL: Class-Incremental Learning
> Incremental Learning: Life-Long Learning/Continual Learning

## 🚀 Survey

| Title | Venue | Year | Code |
| ----- | ----- | ---- | ---- |
| [Class-Incremental Learning: A Survey](http://arxiv.org/abs/2302.03648) | TPAMI | 2024 | [Official](https://github.com/zhoudw-zdw/CIL_Surve) |
| [Continual Learning with Pre-Trained Models: A Survey](http://arxiv.org/abs/2401.16386) | IJCAI | 2024 | [Official](https://github.com/sun-hailong/LAMDA-PILOT) |
| [PyCIL: A Python Toolbox for Class-Incremental Learning](https://arxiv.org/abs/2112.12533) |       |      | [Official](https://github.com/G-U-N/PyCIL)             |

## 🌟 Papers

| Title                                                        | Venue | Year | Type   | Code                                                   |
| ------------------------------------------------------------ | ----- | ---- | ------ | ------------------------------------------------------ |
| [Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning](http://arxiv.org/abs/2403.12030) | CVPR  | 2024 | PTM    | [Official](https://github.com/sun-hailong/CVPR24-Ease) |
| [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](http://arxiv.org/abs/2404.00228) | CVPR  | 2024 | PTM    | [Official](https://github.com/liangyanshuo/InfLoRA)    |
| [CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning](http://arxiv.org/abs/2211.13218) | CVPR  | 2023 | PTM    | [Official](https://github.com/GT-RIPL/CODA-Prompt)     |
| [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/abs/2204.04799) | ECCV  | 2022 | PTM    | [Official](https://github.com/google-research/l2p)     |
| [Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654) | CVPR  | 2022 | PTM    | [Official](https://github.com/google-research/l2p)     |
| [FOSTER: Feature Boosting and Compression for Class-Incremental Learning](https://arxiv.org/abs/2204.04662) | ECCV  | 2022 | Mixed  | [Official](https://github.com/G-U-N/ECCV22-FOSTER)     |
| [DER: Dynamically Expandable Representation for Class Incremental Learning](2021) | CVPR  | 2021 | Mixed  | [Official](https://github.com/G-U-N/ECCV22-FOSTER)     |
| [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725) | CVPR  | 2017 | Memory | [Official](https://github.com/srebuffi/iCaRL)          |

## 📝 Reproduced Results

- class split: `B-$m$ Inc-$n$' . $m$ represents the number of categories in the initial incremental task, while $n$ denotes the number of subsequent incremental tasks, with categories in these tasks evenly distributed. If $m = 0$, all categories in the dataset are evenly distributed across $n$ incremental tasks.
- pre-trained backbone: ViT-B/16-IN21K as the 
- log: 'log/'
- accuracy：CNN/NME

### CIFAR-100

|             | B0 Inc5    | B0 Inc10    | B0 Inc20    | B50 Inc5    | B0 Inc10    | Code                 |
| ----------- | ---------- | ----------- | ----------- | ----------- | ----------- | -------------------- |
| Ease        |            |             |             |             |             |                      |
| InfLoRA     |            |             |             |             |             |                      |
| CODA-Prompt |            |             |             |             |             |                      |
| DualPrompt  | 91.4       | 90.34       | 88.51       | 87.48       | 80.99       | `📁 LAMDA-PILOT-main` |
| L2P         | 90.81      | 89.86       | 87.58       | 87.61       | 79.42       | `📁 LAMDA-PILOT-main` |
| FOSTER      |            |             |             |             |             |                      |
| DER         |            |             |             |             |             |                      |
| iCaRL       | 87.34/90.9 | 85.74/90.22 | 84.07/89.43 | 83.46/88.83 | 80.67/87.06 | `📁 LAMDA-PILOT-main` |



### CUB-200

### ImageNet-A

### Omnibenchmark

### VTAB

## 👨‍🏫  TODO

| Title                                                        | Venue | Year | Type | Code                                                    |
| ------------------------------------------------------------ | ----- | ---- | ---- | ------------------------------------------------------- |
| [Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need](https://arxiv.org/pdf/2303.07338) |       |      |      | [Official](https://github.com/zhoudw-zdw/RevisitingCIL) |
|                                                              |       |      |      |                                                         |
|                                                              |       |      |      |                                                         |



## 🤗Acknowledgments

- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
- [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
