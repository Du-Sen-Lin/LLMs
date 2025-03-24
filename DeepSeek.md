# DeepSeek

DeepSeek应用在工业质检？

Ollama: **本地大语言模型（LLM）运行平台**，允许用户 **在本地设备（如 PC、Mac）上运行和管理 AI 大模型**，而无需依赖云计算。**本地运行 AI 并保障数据隐私**.

## DeepSeekV3(2024):

```
Abstract:
	我们提出了DeepSeek-V3，这是一个强大的专家（MOE）语言模型，其总参数为671b，每个令牌都激活了37B。为了获得有效的推理和具有成本效益的培训，DeepSeek-V3采用了多头潜在注意力（MLA）和DeepSeekmoe架构，这些架构在DeepSeek-V2中得到了彻底验证。此外，DeepSeek-V3先驱者是一种无辅助损失策略，用于负载平衡，并为更强的性能设定了多句话的预测训练目标。我们将DeepSeek-v3预先训练14.8万亿多种和高质量的标记，然后进行了监督的微调和强化学习阶段，以充分利用其能力。全面的评估表明，DeepSeek-V3的表现优于其他开源模型，并实现了与领先的封闭源模型相当的性能。尽管表现出色，但DeepSeek-V3仅需278.8万H800 GPU小时才能进行全面培训。此外，其训练过程非常稳定。在整个培训过程中，我们没有遇到任何无法抵消的损失尖峰或进行任何回滚。该模型检查点可在https://github.com/deepseek-ai/deepseek-v3上找到.
```

blog参考：https://blog.csdn.net/qq_41185868/article/details/144756016

论文阅读：https://yunyaniu.blog.csdn.net/article/details/145331021

创新点：

● 采用DeepSeek-V2中验证有效的Multi-head Latent Attention (MLA) 和 DeepSeekMoE 架构，以提高推理效率和降低训练成本。
● 首次提出辅助损失免费的负载均衡策略 (Auxiliary-loss-free strategy)，最大程度减少负载均衡对模型性能的负面影响。
● 采用多token预测训练目标 (Multi-token prediction training objective)，提升模型在评估基准上的整体性能。

训练过程：包含预训练、监督微调和强化学习三个阶段，在14.8万亿高质量和多样化token上进行预训练。整个训练过程稳定，没有出现不可恢复的损失峰值或回滚。

性能：优于其他开源模型，与领先的闭源模型性能相当，且训练成本低廉（278.8万H800 GPU小时，约合557.6万美元）。

Infrastructures（基础设施）：

```
1、Compute Clusters计算集群：硬件配置(采用2048个H800 GPU)、节点内部互联(每个节点包含8个通过NVLink和NVSwitch互连的GPU)、节点间互联(节点间使用InfiniBand (IB) 互连)

2、Training Framework训练框架：
	框架和并行策略：使用高效轻量级的HAI-LLM训练框架，采用16路流水线并行 (PP)、跨8个节点的64路专家并行 (EP) 和ZeRO-1数据并行 (DP)。
	工程优化：DualPipe算法实现高效PP算法+高效的跨节点全对全通信内核(充分利用InfiniBand和NVLink带宽)+极度节省内存(重新计算RMSNorm和MLA上投影+在CPU中保存EMA参数+共享MTP模块和主模型的嵌入层和输出头等)

3、FP8 Training训练：基于FP8的混合精度框架+细粒度量化+提高累加精度+低精度存储和通信
4、Inference and Deployment推理与部署：将预填充和解码阶段分开部署
5、Suggestions on Hardware Design关于硬件设计的建议：硬件厂商(开发卸载通信任务的协处理器+提高FP8 GEMM累加精度+支持tile和block级量化+支持转置GEMM操作)
```



### Architecture: 

Innovative Load Balancing Strategy and Training Objective架构：创新的负载均衡策略与训练目标

Basic Architecture基本架构：基于Transformer框架+MLA高效推理+DeepSeekMoE高效训练+ALFLB实现负载均衡+CSWAL补充的序列级辅助损失+NLR降低训练过程中的通信成本+NTD策略

- Multi-Head Latent Attention多头潜在注意力

- DeepSeekMoE with Auxiliary-Loss-Free Load Balancing具有无辅助损失负载均衡的 DeepSeekMoE：采用DeepSeekMoE以降低训练成本
- Multi-Token Prediction多标记预测：引入MTP训练目标提升模型性能——扩展预测范围+MTP模块保持每个预测深度的完整因果链+对每个预测深度计算交叉熵损失+推理中丢弃MTP

Multi-Token Prediction多标记预测：引入MTP训练目标提升模型性能——扩展预测范围+MTP模块保持每个预测深度的完整因果链+对每个预测深度计算交叉熵损失+推理中丢弃MTP

- MTP Modules模块实现：使用多个顺序模块来预测多个额外token，保持每个预测深度的完整因果链
- MTP in Inference推理中的 MTP：推理过程中可以丢弃MTP模块，或将其用于推测性解码以提高生成速度

### Pre-Training: 

Towards Ultimate Training Efficiency预训练：迈向极致训练效率

- Data Construction数据构建：优化预训练语料库=提高数学和编程样本比例+扩展多语言+文档打包+FIM策略
  - 语料库优化：提高数学和编程样本的比例+扩展多语言
  - 文档打包(数据完整性)→14.8T(高质量且多样化)
  - Fill-in-Middle (FIM) 策略：沿用DeepSeekCoder-V2中的FIM策略的PSM框架+文档级别
  - 分词器：采用BPE +词汇表(128K)+随机拆分

- Hyper-Parameters超参数：模型超参数（Transformer层数、隐藏维度、注意力头数等）和训练超参数（优化器、学习率调度、批量大小等）
- Long Context Extension长上下文扩展：沿用YaRN方法(仅应用于解耦共享键)+2个额外的训练阶段(4K→32K→128K，每个阶段包含1000步)，NIAH测试良好
- 评估 (Evaluations): 在多个英语、中文和多语言基准上对DeepSeek-V3进行评估，包括知识、代码、数学和推理等方面。
- 讨论 (Discussion): 进行了多token预测策略和辅助损失免费负载均衡策略的消融实验，并分析了批量级负载均衡与序列级负载均衡的区别。

### Post-Training: 

Knowledge Distillation from DeepSeek-R1后训练：从 DeepSeek-R1 中的知识蒸馏； DeepSeek-V3模型的后训练阶段，包括监督微调、强化学习以及相应的评估和讨论。

- Supervised Fine-Tuning后训练处理：采用150万个指令微调数据集
  - 数据集构建：构建包含150万个样本的指令微调数据集，涵盖多个领域，每个领域采用不同的数据创建方法
  - SFT设置：2轮迭代微调+余弦衰减策略+每个序列由多个样本打包而成+采用样本掩码策略(确保样本之间相互隔离)
- Reinforcement Learning强化学习：基于规则的奖励模型和基于模型的奖励模型+采用GRPO算法
- Evaluations评估：标准评估、开放式评估
- 讨论 (Discussion): 讨论了从DeepSeek-R1模型蒸馏知识、自奖励和多token预测的评估结果





## DeepSeek R1(2025):

Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, 通过增强学习激励LLM中的推理能力.

```
Abstract:
	我们介绍了第一代推理模型，DeepSeek-R1-Zero和DeepSeek-R1。 DeepSeek-R1-Zero是一种通过大规模增强学习（RL）训练的模型，而无需监督微调（SFT）作为初步的步骤，表现出了显着的推理能力。通过RL，DeepSeek-R1-Zero自然而然地出现了许多强大而有趣的推理行为。但是，它遇到了挑战，例如不良的可读性和语言混合。为了解决这些问题并进一步提高推理性能，我们介绍了DeepSeek-R1，该问题在RL之前结合了多阶段培训和冷启动数据。 DeepSeekr1在推理任务上实现与OpenAI-O1-1217相当的性能。为了支持研究社区，我们开放源DeepSeek-R1-Zero，DeepSeek-R1和六个密集的型号（1.5b，7b，8b，8b，14b，32b，32b，70b），根据Qwen和Llama蒸馏出了DeepSeek-R1。
```

resource:

```python
# blog:
https://yunyaniu.blog.csdn.net/article/details/145331014
https://yunyaniu.blog.csdn.net/article/details/145293767
```

核心贡献：

- **后训练**：直接在基础模型上应用大规模强化学习，开发了DeepSeek-R1-Zero，展示了LLMs通过纯[RL](https://so.csdn.net/so/search?q=RL&spm=1001.2101.3001.7020)自我进化的潜力
- **蒸馏**：展示了将大模型的推理模式蒸馏到小模型中的有效性，显著提升了小模型的推理能力





