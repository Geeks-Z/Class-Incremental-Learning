## Awesome-CIL

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" style="zoom: 60%;" /></div>
<p></p>
<div align=center><img src="https://visitor-badge.laobi.icu/badge?page_id=Geeks-Z.Awesome-CIL&left_color=green&right_color=red" /> <img src="https://img.shields.io/github/last-commit/Geeks-Z/Awesome-CIL" /> <img src="https://img.shields.io/github/license/Geeks-Z/Awesome-CIL" /></div>

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
- log: 'LAMDA-PILOT-main/res'
- accuracy：CNN/NME
- code: `📁 LAMDA-PILOT-main`

### CIFAR-100

|             | B0 Inc5     | B0 Inc10    | B0 Inc20    | B50 Inc5    | B50 Inc10   |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Ease        | 93.11       | 92.56       | 91.61       | 90.35       | 89.2        |
| CODA-Prompt | 92.85       | 91.49       | 88.9        | 86.38       | 77.81       |
| DualPrompt  | 91.4        | 90.34       | 88.51       | 87.48       | 80.99       |
| L2P         | 90.81       | 89.86       | 87.58       | 87.61       | 79.42       |
| FOSTER      | 91.25/93.05 | 91.56/92.9  | 92.02/92.75 | 90.56/91.91 | 90.82/91.65 |
| DER         | 88.78/91.36 | 88.58/91.01 | 88.57/90.85 | 86.83/89.38 | 86.56/88.98 |
| iCaRL       | 87.34/90.9  | 85.74/90.22 | 84.07/89.43 | 83.46/88.83 | 80.67/87.06 |
| Simplecil   | 81.12       | 82.31       | 82.79       | 78.66       | 78.54       |
| Finetune    | 81.48       | 76.93       | 72.14       | 82.2        | 79.99       |

### CUB-200

> fixed_memory: True

|             | B0 Inc5     | B0 Inc10    | B0 Inc20    | B100 Inc5   | B100 Inc10  |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Ease        | 90.2        | 90.19       | 91.34       | 85.29       | 87.54       |
| CODA-Prompt | 85.65       | 85.01       | 82.92       | 79.6        | 76.63       |
| DualPrompt  | 85.48       | 84.87       | 82.96       | 77.06       | 73.87       |
| L2P         | 83.79       | 81.76       | 79.32       | 79.4        | 74.62       |
| FOSTER      | 78.55/90.58 | 80.7/91.05  | 86.03/91.89 | 82.98/89.17 | 85.55/88.7  |
| DER         | 89.73/89.82 | 88.1/88.93  | 90.09/91.02 | 87.28/88.7  | 86.76/88.49 |
| iCaRL       | 87.42/89.12 | 88.27/90.13 | 88.07/89.9  | 85.9/88.11  | 85.47/87.53 |
| Simplecil   | 89.93       | 90.58       | 90.97       | 87.56       | 87.48       |
| Finetune    | 78.05       | 71.38       | 61.57       | 73.38       | 63.81       |

### ImageNet-R

|             | B0 Inc5 | B0 Inc10 | B0 Inc20 | B100 Inc5 | B100 Inc10 |
| ----------- | ------- | -------- | -------- | --------- | ---------- |
| Ease        | 81.6    | 80.92    | 78.9     | 77.95     | 76.59      |
| CODA-Prompt | 79.99   | 78.63    | 74.27    | 73.15     | 67.83      |
| DualPrompt  | 73.51   | 70.97    | 69.92    | 64.82     | 59.55      |
| L2P         | 76.85   | 76.26    | 73.7     | 69.7      | 64.89      |
| Simplecil   | 65.89   | 67.09    | 67.6     | 63.55     | 63.43      |
| Finetune    | 71.64   | 68.97    | 64.41    | 70.76     | 67.11      |

### ImageNet-A

|             | B0 Inc5 | B0 Inc10 | B0 Inc20 | B100 Inc5 | B100 Inc10 |
| ----------- | ------- | -------- | -------- | --------- | ---------- |
| Ease        | 67.96   | 62.58    | 57.68    | 63.06     | 62.19      |
| CODA-Prompt | 61.0    | 56.06    | 47.77    | 56.61     | 52.92      |
| DualPrompt  | 54.01   | 52.47    | 48.44    | 46.87     | 44.59      |
| L2P         | 53.24   | 52.57    | 45.68    | 49.12     | 46.74      |
| Simplecil   | 58.09   | 59.33    | 60.05    | 53.29     | 53.21      |
| Finetune    | 38.53   | 32.03    | 7.28     | 31.16     | 24.54      |

### Omnibenchmark

|             | B0 Inc5     | B0 Inc10    | B0 Inc20    | B0 Inc30    | B150 Inc5   | B150 Inc10  |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Ease        |             |             |             |             |             |             |
| CODA-Prompt |             |             |             |             |             |             |
| DualPrompt  |             |             |             |             |             |             |
| L2P         |             |             |             |             |             |             |
| FOSTER      |             |             |             |             |             |             |
| DER         | 77.49/81.15 | 77.53/80.74 | 77.21/80.26 |             |             |             |
| iCaRL       | 76.8/82.5   | 75.26/80.94 | 73.38/79.78 | 73.94/79.77 | 74.39/79.13 | 72.23/78.41 |
| Simplecil   | 77.73       | 78.51       | 79.18       | 79.46       | 74.47       | 74.32       |
| Finetune    | 70.21       | 65.12       | 60.77       | 57.2        | 68.37       | 64.1        |

### VTAB

|             | B0 Inc5 | B0 Inc10 |
| ----------- | ------- | -------- |
| Ease        | 90.47   | 91.07    |
| CODA-Prompt | 87.82   | 83.36    |
| DualPrompt  | 89.96   | 87.61    |
| L2P         | 82.11   | 78.04    |
| Simplecil   | 90.94   | 91.66    |
| Finetune    | 78.69   | 59.2     |

## 👨‍🏫  TODO

| Title                                                        | Venue | Year | Type | Code                                                    |
| ------------------------------------------------------------ | ----- | ---- | ---- | ------------------------------------------------------- |
| [Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need](https://arxiv.org/pdf/2303.07338) |       |      |      | [Official](https://github.com/zhoudw-zdw/RevisitingCIL) |
|                                                              |       |      |      |                                                         |
|                                                              |       |      |      |                                                         |



## 🤗Acknowledgments

- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
- [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
