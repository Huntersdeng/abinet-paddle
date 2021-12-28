# Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition
作者: S. Fang, H. Xie, Y. Wang, Z. Mao and Y. Zhang


A paddle implementation for [**ABINet**](https://arxiv.org/abs/2103.06495) (CVPR 2021, Oral).

## 目录

```
1. 简介
2. 数据集和复现精度
3. 开始使用
4. 代码结构与详细说明
```

## 1. 简介
ABINet使用一个视觉模型和一个显示语言模型来识别场景文字，并且可以端到端地训练。语言模型（BCN）模拟了完形填空式的双向语言模型。另外，该语言模型使用了迭代式的文本修正策略。

<img src="figs/doc/overview.bmp" style="height:300;">

## 2.数据集和复现精度
Evaluation datasets, LMDB datasets can be downloaded from [BaiduNetdisk(passwd:1dbv)](https://pan.baidu.com/s/1RUg3Akwp7n8kZYJ55rU5LQ), [GoogleDrive](https://drive.google.com/file/d/1dTI0ipu14Q1uuK4s4z32DqbqF3dJPdkk/view?usp=sharing).

    1. ICDAR 2013 (IC13)
    2. ICDAR 2015 (IC15)
    3. IIIT5K Words (IIIT)
    4. Street View Text (SVT)
    5. Street View Text-Perspective (SVTP)
    6. CUTE80 (CUTE)

|IC13|SVT|IIIT|IC15|SVTP|CUTE|AVG|
|-|-|-|-|-|-|-|
|97.0|93.4|96.4|85.9|89.5|89.2|92.7|## 3. 开始使用

### 3.1 准备环境

- 框架：
  - PaddlePaddle == 2.2.1
  - PaddleOCR == 2.4

- 克隆本项目：

      git clone https://github.com/Huntersdeng/abinet-paddle.git
      cd abinet-paddle

- 安装第三方库：

      pip install -r requirements.txt


### 3.2 快速开始

* **模型验证:**（需要首先下载数据集，并在配置文件中修改数据集的路径）

        python tools/eval.py -c configs/rec/rec_r45_abinet.yml -o Global.pretrained_model='your model path'

  * **模型推断并可视化结果:**（需要在配置文件中将“infer_img”字段修改为预测图片的路径）

        python tools/infer_rec.py -c configs/rec/rec_r45_abinet.yml -o Global.pretrained_model='your model path'