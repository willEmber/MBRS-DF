# MBRS-DF: MBRS for Deepfake Detection

## MBRS：通过真实和模拟JPEG压缩的小批量处理增强DNN水印的鲁棒性 (MBRS-DF 变体)

原始作者: Zhaoyang Jia, Han Fang, Weiming Zhang (来自中国科学技术大学)
MBRS-DF 修改者: [stdlibh]

原始论文: [[arXiv]](https://arxiv.org/abs/2108.08211) [[PDF]](https://arxiv.org/pdf/2108.08211)

> 这是论文 *MBRS：通过真实和模拟JPEG压缩的小批量处理增强DNN水印的鲁棒性* 的源代码的一个修改版本，旨在使水印对 Deepfake 操作敏感，同时保持对常见良性操作的鲁棒性。此变体称为 MBRS-DF。如果您发现任何bug，请在 *issue* 页面或通过邮件联系我。谢谢！

****
### MBRS-DF 核心思想

MBRS-DF 引入了以下关键修改：

1.  **Deepfake 模拟噪声层**: 添加了多种模拟 Deepfake 操作效果的噪声层，例如：
    *   `DeepfakeProxy`: 使用轻量级自编码器模拟重建过程。
    *   `RegionalDestruction`: 模拟对图像特定区域（如人脸）的破坏。
    *   `FrequencyModulation`: 模拟 Deepfake 可能引入的频率特征变化。
2.  **条件化损失函数**: 在训练过程中，根据当前应用的噪声类型动态调整损失函数：
    *   **良性操作 (如 JPEG, 模糊)**: 训练解码器以最小化消息恢复错误率，增强鲁棒性。
    *   **Deepfake 模拟操作**: 训练解码器以最大化消息恢复错误率（目标是使水印失效），增强脆弱性/敏感性。
3.  **混合噪声训练**: 在训练期间，随机从良性操作和 Deepfake 模拟操作中选择噪声层，使模型同时学习鲁棒性和脆弱性。

目标是生成一种水印，它能抵抗常规图像处理，但在图像被 Deepfake 修改后会被破坏，从而提供一种检测 Deepfake 的信号。

****
### 2021/10/03更新：扩散模型的训练 (原始 MBRS)

由于使用扩散模型的模型训练过程（关于扩散模型的详细信息请参阅[论文](https://arxiv.org/pdf/2108.08211)）不够稳定，我们更新了训练过程以获得更稳定的效果。

- **原始训练**适用于128x128图像和30位消息，经过全连接嵌入后为256维。批次大小为16，学习率为1e-3。我们训练了300个周期，并根据验证结果在第110个周期应用早停策略。通过这种方式，我们获得了预训练模型，[论文](https://arxiv.org/pdf/2108.08211)中的结果就是基于此。

- 然而，这种训练过程对于裁剪鲁棒性([Crop attack · Issue #2](https://github.com/jzyustc/MBRS/issues/2))不够稳定，也就是说，对于*Crop(p=3.5%)*，裁剪的验证结果在BER=2%到BER=25%之间变化，很难保证我们得到的良好结果。

- 为了解决这个问题，我们以一种简单但有效的方式**更新了训练过程**。
  - 首先，我们像**原始训练**一样训练模型100个周期，并在第92个周期应用早停策略，获得一个次优模型（对于*Crop(p=3.5%)*，BER = 20%，PSNR=29.75）。
  - 然后，我们使用相同的设置但学习率=1e-4对模型进行50个周期的微调，并在第13个周期应用早停策略，获得最优模型（对于*Crop(p=3.5%)*，BER = 1.85%，PSNR=30.89）。

希望这能帮到你 :)


****

### 环境要求

我们在开发此项目时使用了以下软件包/版本。 (MBRS-DF 未引入新的核心依赖)

- Pytorch `1.5.0`
- torchvision `0.6.0` (注意：原始文档中 torchvision 版本有误，根据环境配置部分应为 0.6.0)
- kornia `0.3.0`
- numpy `1.16.4`
- Pillow `6.0.0`
- scipy `1.3.0`
- matplotlib (用于可视化)
- tqdm (用于进度条)

****

### 数据集准备

请下载 ImageNet 或 COCO 数据集，并将它们放入 `datasets` 文件夹，如下所示：

```
├── datasets
│   ├── train
│   │   ├── xxx.jpg
│   │   ├── ...
│   ├── test
│   │   ├── xxx.jpg
│   │   ├── ...
│   ├── validation
│   │   ├── xxx.jpg
│   │   ├── ...
├── ...
├── results
```

有关所使用数据集的更多详细信息，请阅读原始[论文](https://arxiv.org/pdf/2108.08211)。



### 预训练模型

请在[Google Drive](https://drive.google.com/drive/folders/1A_SAqvU2vMsHxki0s9m9rKa-g8B6aghe?usp=sharing)下载预训练模型，并将它们放在`results/xxx/models/`路径下。（xxx是项目名称，例如MBRS_256_m256）


****

### 环境配置指南

以下是使用Conda创建虚拟环境并配置项目的详细步骤：

1. **安装Anaconda或Miniconda**：
   - 如果尚未安装，请从[Anaconda官网](https://www.anaconda.com/products/distribution)或[Miniconda官网](https://docs.conda.io/en/latest/miniconda.html)下载并安装

2. **创建虚拟环境**：
   ```bash
   # 创建名为mbrs的环境，指定Python版本为3.7
   conda create -n mbrs python=3.7
   # 激活环境
   conda activate mbrs
   ```

3. **安装PyTorch和torchvision**：
   ```bash
   # 安装指定版本的PyTorch和torchvision
   conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
   # 注意：如果您没有GPU，请使用以下命令
   # conda install pytorch==1.5.0 torchvision==0.6.0 cpuonly -c pytorch
   ```

4. **安装其他依赖项**：
   ```bash
   # 安装其他必要的软件包
   pip install kornia==0.3.0
   conda install numpy==1.16.4 pillow==6.0.0 scipy==1.3.0
   pip install matplotlib tqdm
   ```

5. **克隆代码库**（如果您尚未下载）：
   ```bash
   git clone https://github.com/jzyustc/MBRS.git
   cd MBRS
   ```

### 运行项目

1. **数据集准备**：
   - 按照前面"数据集准备"部分的说明准备数据集
   - 确保数据集结构正确

2. **下载预训练模型**（可选）：
   - 如果您想跳过训练过程，请下载预训练模型
   - 将模型文件放在`results/xxx/models/`目录中

3. **配置训练参数**：
   - 编辑`train_settings.json`文件，根据您的需求调整参数
   - 主要参数包括：
     ```json
     {
       "project_name": "MBRS_256_m256",
       "train_path": "./datasets/train",
       "validation_path": "./datasets/validation",
       "image_size": 128,
       "message_length": 30,
       "batch_size": 16,
       "learning_rate": 1e-3,
       "epochs": 100
     }
     ```

4. **开始训练**：
   ```bash
   python train.py
   ```
   - 对于微调，可以修改`train_settings.json`中的学习率为1e-4，并设置`load_model: true`
   - 重新运行训练：
     ```bash
     python train.py
     ```

5. **测试模型**：
   - 编辑`test_settings.json`文件，设置测试参数
   - 运行测试脚本：
     ```bash
     python test.py
     ```

6. **查看结果**：
   - 训练和测试结果将保存在`results/xxx/`目录中
   - 检查日志文件了解性能指标
   - 查看生成的水印图像和提取结果

### 常见问题解决

1. **CUDA错误**：
   - 确认您的CUDA版本与PyTorch兼容
   - 尝试使用`nvidia-smi`命令检查GPU状态
   - 如果出现内存不足错误，尝试减小批次大小

2. **数据集加载问题**：
   - 确保数据集路径正确
   - 验证图像格式是否支持（JPG/PNG）
   - 检查是否有损坏的图像文件

3. **训练不稳定**：
   - 按照更新后的训练过程进行（先训练后微调）
   - 尝试调整学习率和批次大小
   - 确保早停机制正常工作

如有更多问题，请查看GitHub仓库的Issues页面或直接联系作者。

****

### 训练

在json文件`train_settings.json`中更改设置，然后运行：

```bash
python train.py
```

日志文件和结果将保存在`results/xxx/`中



### 测试

在json文件`test_settings.json`中更改设置，然后运行：

```bash
python test.py
```

日志文件和结果将保存在`results/xxx/`中


****


### 引用

如果您发现此代码库有用，请引用我们的论文！

```
@inproceedings{jia2021mbrs,
  title={MBRS: Enhancing Robustness of DNN-based Watermarking by Mini-Batch of Real and Simulated JPEG Compression},
  author={Zhaoyang  Jia, Han Fang and Weiming Zhang},
  booktitle={arXiv:2108.08211},
  year={2021}
}
```





联系方式: [jzy_ustc@mail.ustc.edu.cn](mailto:jzy_ustc@mail.ustc.edu.cn)

