# LoRA

# 一、介绍：

 LoRA (Low-Rank Adaptation)

**LoRA**（Low-Rank Adaptation of Large Language Models）是一种参数高效的微调方法，特别用于减少微调大型模型所需的存储和计算资源。LoRA 的核心思想是，通过在预训练的权重上添加一个低秩矩阵来表示微调中的变化，达到在不更改原始模型权重的情况下高效适配新任务的效果。

### LoRA的主要特点和工作原理

1. **低秩矩阵分解**：LoRA 假设模型中某些权重矩阵的变化是低秩的，因此在微调过程中，只需要学习一个低秩矩阵（比原矩阵小很多），这大大减少了参数量。
2. **冻结预训练权重**：在微调时，LoRA 会冻结原始模型的权重，仅学习和更新新增的低秩矩阵。通过这样的设计，LoRA 能够保持模型的预训练能力，避免过拟合，同时加快微调速度。
3. **更低的存储成本**：由于 LoRA 不会更新原始的权重矩阵，只增加少量的低秩矩阵，因此存储和部署的成本远低于传统的微调方法。
4. **适用于多种生成任务**：LoRA 可应用于各种生成任务，比如文本生成、图像生成、对话生成等，因为它仅修改权重矩阵，不受特定生成模型架构的限制。

### LoRA的应用

LoRA 主要用于以下场景：

- **大语言模型的微调**：在 NLP 中，通过 LoRA 微调大型语言模型，能有效地适配于下游任务。
- **图像生成模型的微调**：在图像生成中，比如扩散模型中，LoRA 可以让模型在不改变主干结构的情况下微调到新风格或新类别的数据上。

以下是一个具体使用 LoRA (Low-Rank Adaptation) 的实例，示范如何在 Hugging Face 的 Transformers 库中对预训练的大型语言模型进行微调。这个示例主要集中在文本生成任务上。

1. 环境准备

确保你已经安装了 Hugging Face 的 `transformers` 和 `peft`（用于 LoRA）库。如果还没有安装，可以通过以下命令安装：

```
pip install transformers peft
```

2. 导入必要的库

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
```

3. 加载预训练模型和分词器

在这个例子中，我们使用 `gpt2` 模型作为基础模型：

```
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

4. 配置 LoRA

设置 LoRA 的参数，如低秩适应的维度：

```
lora_config = LoraConfig(
    r=16,  # 低秩矩阵的秩
    lora_alpha=32,  # 调整学习率的超参数
    lora_dropout=0.1,  # dropout 比例
    task_type=TaskType.CAUSAL_LM  # 任务类型，因是生成任务用 CAUSAL_LM
)

# 包装模型以使用 LoRA
lora_model = get_peft_model(model, lora_config)
```

5. 数据准备

假设你已经有了训练数据，可以是简单的文本文件。这里我们创建一些示例数据：

```
train_texts = [
    "The cat sits on the mat.",
    "Dogs are great companions.",
    "The sun rises in the east.",
    "I love to read books about space."
]
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
```

6. 微调模型

使用 PyTorch 的 `DataLoader` 来迭代训练数据并微调模型：

```
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW

# 创建数据集
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'])
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 定义优化器
optimizer = AdamW(lora_model.parameters(), lr=5e-5)

# 微调
lora_model.train()
for epoch in range(3):  # 设置 epoch 数量
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask = batch
        outputs = lora_model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")
```

7. 生成文本

微调完成后，你可以使用微调后的模型生成文本：

```
lora_model.eval()
input_prompt = "The future of AI is"
input_ids = tokenizer(input_prompt, return_tensors='pt').input_ids

with torch.no_grad():
    generated_ids = lora_model.generate(input_ids, max_length=50)
    
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)
```

## 二、“Base + PEFT” (Parameter-Efficient Fine-Tuning) 的范式

在 Base + PEFT 范式中：

1. **Base Model (基础模型)**：使用已经预训练的大模型，例如稳定扩散模型 (Stable Diffusion)。
2. **PEFT (参数高效微调)**：通过减少训练参数量，增加训练效率并降低资源消耗。LoRA 是 PEFT 的一种，它通过为模型添加小规模的可微调参数模块而不改变原模型的大量参数，从而实现对新任务的适应性。

例如：“Stable Diffusion + LoRA” 

```python
# LoRA 在 Stable Diffusion 中的应用
# LoRA 的基本思路是：
    参数注入：在基础模型的权重矩阵上添加低秩矩阵来作为新参数，这些矩阵在推理过程中起作用。
    减少训练成本：LoRA仅微调少量的额外参数，不改变原模型的主要权重，因此训练开销小、效率高。
    具体到 Stable Diffusion，LoRA 可以用于在个性化风格上进行生成，生成更加符合特定需求的图像风格。
# Stable Diffusion + LoRA 的流程,以下是实现此组合的典型步骤：
	加载基础模型：加载预训练的 Stable Diffusion 模型。
	添加 LoRA 层：在模型的关键层（通常是自注意力层和卷积层）中添加 LoRA 适配模块。
	微调：用少量的目标数据对 LoRA 层进行微调，使其学会生成特定风格或内容。基础模型的参数不被更新，只对 LoRA 层参数进行优化。
	推理阶段：在推理时，Stable Diffusion 将结合 LoRA 的微调结果进行图像生成，以生成符合特定需求的内容。
```

