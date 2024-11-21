# T2I（文本到图像）

# 前置：

**文本到图像（Text-to-Image, T2I）**技术是一种利用自然语言处理和生成模型来将文本描述转化为图像的生成技术。T2I 技术主要依赖深度学习，尤其是扩散模型、生成对抗网络（GAN）、变分自编码器（VAE）等，能够根据文字提示生成高质量、符合描述的图像。T2I 的应用场景非常广泛，包括广告设计、影视制作、虚拟场景生成等。

```python
# T2I 的关键技术
	扩散模型（Diffusion Models）：扩散模型通过逐步去噪的过程来生成图像，比如 DALL-E 2 和 Stable Diffusion 就是典型的 T2I 扩散模型应用。它们可以将随机噪声逐步变为符合描述的图像。
	生成对抗网络（GAN）：GAN 是一种对抗训练模型，包括生成器和判别器两个部分，生成器尝试生成真实感强的图像，而判别器则判断图像真假。GAN 一度是 T2I 的主流技术，但难以稳定训练。
	条件生成模型（Conditional Generative Models）：这类模型会使用输入的文本描述作为生成图像的条件，比如给生成器一个条件语句来描述要生成的图像内容。扩散模型和 GAN 都可以使用条件生成。

# T2I 的工作流程
文本编码：使用自然语言处理模型（如 BERT、GPT 或 CLIP）对输入的文本进行编码，将文本转换为适用于神经网络的向量。
图像生成：将文本编码作为条件输入到生成模型（如扩散模型或 GAN）中，以逐步生成符合文本描述的图像。
多次迭代优化：模型可能会进行多次生成和优化，在不同阶段添加和减少噪声，最终输出清晰、符合描述的图像。

# A1111
A1111 通常指的是 Automatic1111 Web UI，这是一个非常流行的用于 Stable Diffusion 的图形界面（UI）。这个界面在 GitHub 上开源，提供了方便的工具和插件系统，允许用户使用 Stable Diffusion 模型生成图像并自定义生成参数，且无需直接使用命令行操作。
Automatic1111 Web UI 支持许多扩展，例如 ControlNet、LoRA 等，并且可以加载多种预训练模型，使得生成图像的过程更加灵活和直观。
```

```python
# 阅读参考：https://zhuanlan.zhihu.com/p/348498294
在Autoencoder的基础上，显性的对z的分布p(z)进行建模，使得自编码器成为一个合格的生成模型，我们就得到了Variational Autoencoders，即变分自编码器。

loss = log(p_seta(X)) - DKL(p_seta, q_seta); 即最大化loss=最大化似然 与 最小化后验的KL散度； 
VAE 的训练目标是最小化重构误差和隐空间分布与先验分布的 KL 散度之间的差距，从而学到数据的低维表示。

需求的改进：原版VAE只能采样得到z_i后随机生成数字。然而，更多的时候我们可能会希望模型能够生成我们指定的数字。这就引出了CVAE (Conditional VAE)。假设我们现在的数据集为X，我们现在希望利用它的标注Y来控制生成的结果。在MNIST的场景下，就是我们希望能够告诉Decoder：我现在想生成一个标注为"7"的数字。

==> 传统的生成模型：GAN; VAE:
    生成模型的类型有很多，这里主要介绍变分自编码器(Variational AutoEncoder，VAE)和生成对抗网络(Generative Adversarial Network，GAN)两种。其中，变分自编码器使用了两个神经网络:编码器(Encoder)，用于将输入图像转变为隐含变量(Latent Variable)，这个隐含变量的值需要加上一定的限制，使之服从目标分布;解码器(Decoder)，其作用是输入隐含变量，将隐含变量转变为目标的语句。训练完成的模型只需要使用解码器的部分，通过输入随机产生的隐含变量的值，产生隐含变量对应的图像。而对于生成对抗网络，对应的网络是生成网络(Generator)和判别网络(Discriminator)，生成网络的作用是输入服从一定分布的隐含变量，输出对应的图像;判别网络的作用是给定未知来源的图像，判断这个图像是属于神经网络生成的，还是来源于训练数据集的。对应的损失函数也分为两个，第一个是生成网络生成的图像通过判别网络的损失函数，对于生成网络来说，对应的损失函数要使得判别网络难以判别出生成网络生成的图像是生成的图像(或称为「假图像」);对于判别网络来说，对应的损失函数要尽可能判断出训练数据的图像(或称为「真图像」)，而且需要尽可能判断出生成的图像。生成网络和判别网络的训练是一个对抗过程，通过生成网络和判别网络的交替训练，最后到达模型的平衡状态，即判别网络无法判别出图像的真假，这样得到的生成网络就能通过输入隐含变量，输出对应的生成图像。VAE和 GAN 这两种生成模型各有优缺点，对 VAE来说，优点是可以知道输入图像具体对应的隐含变量，缺点是生成的图像比较模糊;对 GAN 来说，优点是生成的图像比较清晰，缺点是无法得到某一个输入图像对应的隐含变量，而且模型训练容易遇到不收敛和模式塌陷(Mode Collapse)等问题。
    
==> Diffusion:
(1)正向扩散过程：逐步向数据添加噪声，最终将数据分解成接近纯噪声的状态。
(2)反向生成过程：从纯噪声开始，通过多个时间步的反向操作一步步去噪，最终还原到原始数据分布。
学习去噪的逆过程。
LDM(Latent Diffusion Model) 是一种基于 Diffusion 的生成模型，为了降低计算成本和提高生成质量，LDM 将 Diffusion 模型应用在潜在空间中，而不是直接在像素空间中。这时，VAE 的作用是将输入图像映射到潜在空间中，使 Diffusion 模型可以在更小、信息更集中的潜在空间中进行生成，从而减少生成步骤并加快运算速度。
LDM 中，VAE 的作用：
（1）降维与信息提取：将高维数据映射到低维潜在空间，提取主要特征信息。
（2）在潜在空间中生成：Diffusion 模型在低维空间中完成生成过程，再通过 VAE 的解码器将生成的潜在向量还原到原始数据空间。
（3）提高生成效率：因为潜在空间比原始空间小，Diffusion 模型的生成过程计算量更低，但仍能保证较高的图像质量。
==>other:MAE 主要用于自监督学习任务，通过恢复被掩蔽的数据来学习特征，可以用于图像、文本等数据的特征提取。
Masked Autoencoders (MAE),自监督学习中的一种创新模型结构，主要用于图像表征学习。其思想基于 Vision Transformers (ViTs)，通过掩蔽（masking）和重构的方式，在没有标签的数据上预训练模型，使模型能够捕获图像中有意义的特征。 MAE 的核心思想是 掩蔽编码（Masked Encoding）。通过随机掩蔽图像输入的一部分，并要求模型从剩余部分重构被掩蔽的内容，这样可以迫使模型理解和捕获输入数据的高层特征表示。
```



# Paper: 

## Stable Diffusion

”**High-Resolution Image Synthesis with Latent Diffusion Models**“， 使用潜在扩散模型的高分辨率图像合成

```python
# 摘要
"""
通过将图像形成过程分解为去噪自动编码器的顺序应用，扩散模型 (DM) 在图像数据及其他方面实现了最先进的合成结果。此外，他们的公式允许一种指导机制来控制图像生成过程而无需重新训练。然而，由于这些模型通常直接在像素空间中运行，因此强大的 DM 的优化通常会消耗数百个 GPU 天，并且由于顺序评估，推理成本很高。为了在有限的计算资源上进行 DM 训练，同时保持其质量和灵活性，我们将它们应用在强大的预训练自动编码器的潜在空间中。与之前的工作相比，在这种表示上训练扩散模型首次允许在复杂性降低和细节保留之间达到接近最佳的点，从而极大地提高了视觉保真度。 通过将交叉注意力层引入模型架构中，我们将扩散模型转变为强大而灵活的生成器，用于一般条件输入（例如文本或边界框），并且以卷积方式使高分辨率合成成为可能。我们的潜在扩散模型 (LDM) 在图像修复和类条件图像合成方面取得了新的最先进分数，并在各种任务上实现了极具竞争力的性能，包括文本到图像合成、无条件图像生成和超分辨率，与基于像素的 DM 相比，同时显着降低了计算要求.
"""
```



### Stable Diffusion v1：

```python
https://github.com/CompVis/stable-diffusion
"""
Stable Diffusion v1 是指模型架构的特定配置，它使用下采样因子 8 自动编码器以及 860M UNet 和 CLIP ViT-L/14 文本编码器来进行扩散模型。该模型在 256x256 图像上进行预训练，然后在 512x512 图像上进行微调。
"""
```

### Stable Diffusion v2：

```python
https://github.com/Stability-AI/stablediffusion
"""
Stable Diffusion v2 是指模型架构的特定配置，它使用下采样因子 8 自动编码器以及 865M UNet 和 OpenCLIP ViT-H/14 文本编码器来进行扩散模型。 SD 2-v 模型产生 768x768 像素输出。
"""
```

### Stable Diffusion v3：

```
https://stability.ai/news/stable-diffusion-3
```

论文阅读参考： https://jarod.blog.csdn.net/article/details/131018599

- 马尔可夫链（Markov Chain）:

  是一种随机过程，它具有马尔可夫性质。马尔可夫性质是指在给定当前状态的情况下，未来状态的概率分布只依赖于当前状态，而与过去的状态无关。简单来说，系统在 t+1 时刻的状态X(t+1）只与 t 时刻的状态X(t)有关，而与t之前的状态X(t-1),  X(t-2) 无关。



## ControlNet(ICCV2023)

Adding Conditional Control to Text-to-Image Diffusion Models， 向文本到图像扩散模型添加条件控制

Abstract: 

```
我们提出了 ControlNet，这是一种神经网络架构，可将空间调节控制添加到大型预训练文本到图像扩散模型中。 ControlNet 锁定可用于生产的大型扩散模型，并重用其经过数十亿图像预训练的深度且强大的编码层作为强大的骨干来学习一组不同的条件控制。神经架构与“零卷积”（零初始化卷积层）连接，参数从零逐渐增长，并确保没有有害噪声会影响微调。我们使用稳定扩散、使用单个或多个条件、有或没有提示来测试各种条件控制，例如边缘、深度、分割、人体姿势等。我们证明了 ControlNet 的训练对于小型（<50k）和大型（>1m）数据集都是稳健的。大量结果表明 ControlNet 可以促进更广泛的应用来控制图像扩散模型。

```

```python
https://github.com/lllyasviel/ControlNet
# sd web ui --- A1111
https://github.com/AUTOMATIC1111/stable-diffusion-webui
#controlnet扩展 https://github.com/Mikubill/sd-webui-controlnet
```



## CtrLoRA(2024)

**CtrLoRA: An Extensible and Efficient Framework for Controllable Image Generation**, 用于可控图像生成的可扩展且高效的框架

Abstract:

```
最近，大规模扩散模型在文本到图像（T2I）生成方面取得了令人瞩目的进展。为了进一步为这些 T2I 模型配备细粒度的空间控制，ControlNet 等方法引入了一个额外的网络来学习遵循条件图像。然而，对于每种单一条件类型，ControlNet 都需要使用数百个 GPU 小时对数百万个数据对进行独立训练，这非常昂贵，并且对于普通用户探索和开发新类型的条件来说具有挑战性。为了解决这个问题，我们提出了 CtrLoRA 框架，它训练 Base ControlNet 以从多个基本条件学习图像到图像生成的常识，以及特定于条件的 LoRA 以捕获每个条件的不同特征。利用我们预训练的 Base ControlNet，用户可以轻松使其适应新条件，只需 1,000 个数据对和不到一小时的单 GPU 训练即可在大多数场景下获得满意的结果。此外，我们的CtrLoRA与ControlNet相比，可学习参数减少了90%，显着降低了模型权重分配和部署的门槛。对各种类型条件的大量实验证明了我们方法的效率和有效性。代码和模型权重将在 https://github.com/xyfJASON/ctrlora 发布。
```

```
https://github.com/xyfjason/ctrlora
```

- I2I (图像到图像)

目的: T2I 模型很难准确控制布局和姿势等空间细节，因为仅靠文本提示不足以精确传达这些细节。

```python
# ControlNet:  增加了一个接受条件图像的额外网络
# 论文阅读： https://blog.csdn.net/jarodyv/article/details/132739842
ControlNet 能够根据特定类型的条件图像（例如 canny edge）生成图像，从而显著提高可控性。然而，对于每种条件类型，都需要使用大量数据和计算资源从头开始训练一个独立的 ControlNet。引入了一个辅助网络来处理条件图像，并将该网络集成到 Stable Diffusion 模型中。

缺点：为每个单一条件训练一个ControlNet需要大量的数据和时间，造成了相当大的负担。
解决：ControlNet-XS 优化了网络架构以加快训练收敛速度； UniControl 与 Uni-ControlNet 训练一个统一的模型来管理多个条件，大大减少了模型数量。
解决方案的不足: 这两种方法缺乏一种直接、方便用户添加新条件的方式，这限制了它们在实际场景中的实用性。相比之下，我们的方法可以用更少的数据和更少的资源有效地学习新条件。
```

```python
# CtrLoRA
# 论文阅读: https://blog.csdn.net/weixin_42475026/article/details/143306616
受 “Base + PEFT” 范式的启发，我们提出了一个 CtrLoRA 框架，使用户可以方便高效地为自定义类型的条件图像建立控制网络。
```

```python
# CLIP（Contrastive Language-Image Pretraining）和T5（Text-to-Text Transfer Transformer）是两个重要的深度学习模型，分别在视觉和语言处理领域取得了显著的成果。以下是对这两个模型的详细介绍：
# 1、CLIP : Contrastive Language-Image Pretraining
CLIP是一个对比学习模型，旨在通过同时学习文本和图像的表示来增强视觉理解能力。它利用大规模的图像-文本对进行训练，以便模型可以将视觉和语言信息对齐。
CLIP的训练过程包括使用图像和对应文本的对比学习任务，模型学习区分相关的图像和文本对与不相关的对，从而提升其对文本描述的理解能力。

# 2、T5 : Text-to-Text Transfer Transformer
T5是一个基于Transformer架构的预训练模型，旨在将所有文本任务统一为文本到文本的格式。这意味着输入和输出都是文本，模型通过一个统一的框架来处理不同的NLP任务，例如翻译、摘要、问答等。
T5在预训练阶段使用了大规模的文本数据集进行训练，使其能够捕捉丰富的语言特征。
```



# Log:



Nov 18, 2024：

```python
# 4090上玩一玩webui
https://github.com/AUTOMATIC1111/stable-diffusion-webui

# 容器里面 cv_env 跑一跑v2
https://github.com/Stability-AI/stablediffusion


```

