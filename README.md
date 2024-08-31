## Awesome-CIL

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" style="zoom: 60%;" /></div>
<p></p>
<div align=center><img src="https://visitor-badge.laobi.icu/badge?page_id=Geeks-Z.Class-Incremental-Learning&left_color=green&right_color=red" /> <img src="https://img.shields.io/github/last-commit/Geeks-Z/Class-Incremental-Learning" /> <img src="https://img.shields.io/github/license/Geeks-Z/Class-Incremental-Learning" /></div>

## 🎉 介绍

CIL: Class-Incremental Learning/类增量学习
> Life-Long Learning/Continual Learning

## 🚀 Survey

| Title | Venue | Year | Code | Code_Position |
| ----- | ----- | ---- | ---- | ------------- |
| [Class-Incremental Learning: A Survey](http://arxiv.org/abs/2302.03648) | TPAMI | 2024 | [CIL_Surve](https://github.com/zhoudw-zdw/CIL_Surve) |`📁CIL_Survey-main`              |
| [Continual Learning with Pre-Trained Models: A Survey](http://arxiv.org/abs/2401.16386) |       | 2024 | [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT) |               |
|       |       |      |      |               |

## 🌟 Papers

| Title                                                        | Venue | Year | Type    | Code                                                    | Code_Position                                   |
| ------------------------------------------------------------ | ----- | ---- | ------- | ------------------------------------------------------- | ----------------------------------------------- |
| [Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning](http://arxiv.org/abs/2403.12030) | CVPR  | 2024 | Adapter | [Official](https://github.com/sun-hailong/CVPR24-Ease)  | `📁 CVPR24-Ease-main `                           |
| [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](http://arxiv.org/abs/2404.00228) | CVPR  | 2024 | LoRA    | [Official](https://github.com/liangyanshuo/InfLoRA)     | `📁 InfLoRA-main`                                |
| [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/abs/2204.04799) | ECCV  | 2022 | Prompt  | [Official](https://github.com/google-research/l2p)      | `📁 dualprompt-pytorch-master`                   |
| [FOSTER: Feature Boosting and Compression for Class-Incremental Learning](https://arxiv.org/abs/2204.04662) | ECCV  | 2022 | Prompt  | [Official](https://github.com/G-U-N/ECCV22-FOSTER)      | `📁 ECCV22-FOSTER-master`                        |
| [PILOT: A Pre-Trained Model-Based Continual Learning Toolbox](https://arxiv.org/abs/2309.07117) |       |      | Toolbox | [Official](https://github.com/sun-hailong/LAMDA-PILOT)  | `📁 LAMDA-PILOT-main`                            |
| [Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654) | CVPR  | 2022 |         | [Official](https://github.com/google-research/l2p)      | `📁 Learn-to-prompt-for-Continual-Learning-main` |
| [PyCIL: A Python Toolbox for Class-Incremental Learning](https://arxiv.org/abs/2112.12533) |       |      | Toolbox | [Official](https://github.com/G-U-N/PyCIL)              | `📁 PyCIL-master`                                |
| [Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need](https://arxiv.org/pdf/2303.07338) |       |      |         | [Official](https://github.com/zhoudw-zdw/RevisitingCIL) | `📁 RevisitingCIL-main`                          |

## 📝 Reproduced Results

- `B-$m$ Inc-$n$' stands for the class split. $m$ represents the number of categories in the initial incremental task, while $n$ denotes the number of subsequent incremental tasks, with categories in these tasks evenly distributed. If $m = 0$, all categories in the dataset are evenly distributed across $n$ incremental tasks.
- ViT-B/16-IN21K as the pre-trained backbone
- 实验日志位于每个方法文件夹下的`res`目录
- results：CNN

### CIFAR-100

#### PEFT

|             | B0 Inc5 | B0 Inc10 | B0 Inc20 | B50 Inc5 | B0 Inc10 | Code                 |
| ----------- | ------- | -------- | -------- | -------- | -------- | -------------------- |
| Ease        |         |          |          |          |          |                      |
| InfLoRA     |         |          |          |          |          |                      |
| CODA-Prompt |         |          |          |          |          |                      |
| DualPrompt  | 91.4    | 90.34    | 88.51    | 87.48    | 80.99    | `📁 LAMDA-PILOT-main` |
| L2P         | 90.81   | 89.86    | 87.58    | 87.61    | 79.42    | `📁 LAMDA-PILOT-main` |

#### Memory

|        | B0 Inc5    | B0 Inc10    | B0 Inc20    | B50 Inc5    | B0 Inc10    | Code                 |
| ------ | ---------- | ----------- | ----------- | ----------- | ----------- | -------------------- |
| FOSTER |            |             |             |             |             | `📁 LAMDA-PILOT-main` |
| DER    |            |             |             |             |             | `📁 LAMDA-PILOT-main` |
| iCaRL  | 87.34/90.9 | 85.74/90.22 | 84.07/89.43 | 83.46/88.83 | 80.67/87.06 | `📁 LAMDA-PILOT-main` |

### CUB-200

### ImageNet-A

### Omnibenchmark

### VTAB





## 👨‍🏫 Acknowledgments
