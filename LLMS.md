# LLMs

Large Language Model，大型语言模型， 如GPT-3、T5、BERT等。

## 一、前置

```python
# https://blog.csdn.net/2401_84204207/article/details/139645338
# bert微调：https://blog.csdn.net/FrenzyTechAI/article/details/131958410
# gpt微调：https://blog.csdn.net/FrenzyTechAI/article/details/131918889
```

### embedding

"embedding"（嵌入）是一个关键概念，它用于将输入数据（如单词、字符或其他类型的标记）转换为模型可以处理的数值向量形式。这些向量捕捉了输入数据的语义和结构信息，使得模型能够在后续的处理中有效地使用这些信息。

**词嵌入（Word Embedding）**

在自然语言处理（NLP）中，词嵌入是最常用的一种embedding形式。它将词汇表中的每个词映射到一个高维空间中的点。这些向量表示能够捕捉词与词之间的语义关系，例如同义词往往在向量空间中更接近。

- **训练获得**：词嵌入可以通过在大规模语料上预训练得到，如Word2Vec、GloVe等模型，也可以在特定任务的训练过程中继续调整和优化。
- **维度**：嵌入向量的维度是一个超参数，需要根据任务和模型大小来调整。一般来说，维度越高，模型能捕捉的信息越丰富，但计算成本也越高。

**位置嵌入（Position Embedding）**

Transformer模型不使用循环或卷积结构，因此无法直接从序列的顺序中获取位置信息。为了使模型能够理解单词在句子中的位置关系，Transformer引入了位置嵌入。

- **作用**：位置嵌入向量与词嵌入向量相加，为模型提供位置信息。这使得模型能够理解词序的重要性。
- **实现**：位置嵌入通常使用正弦和余弦函数生成，使得模型可以通过向量间的相对位置来推断单词间的顺序关系。

### prompt

"Prompt"在NLP和机器学习中是一个多用途的术语，通常涉及到向模型提供输入，以引导其行为。

### bert与gpt

BERT是一个编码器（encoder）模型，主要用于理解文本的上下文; 对于问答（QA）任务，通常需要一段文本作为上下文来寻找答案。如果没有提供上下文文本，BERT无法执行其设计的任务——即从给定的文本中抽取答案。

==> QA:

1、检索式问答模型（Retrieval-based QA）。这类模型首先使用问题去检索一个大型的文档集合或知识库，然后从检索到的文档中抽取答案。这通常涉及两个主要步骤：

- **文档检索**：使用问题去检索最相关的文档或信息片段。
- **答案抽取**：从检索到的文档中抽取答案。

2、生成式问答模型（Generative QA）。生成式问答模型直接生成答案，而不是从现有文本中抽取。这通常需要一个预训练的大型语言模型（如GPT-3、BERT的解码器版本等），这些模型能够根据问题生成连贯的答案。

==> QA:

1、 **BERT-based QA Models**:

你可以微调标准的BERT模型或BERT的变体，常用于阅读理解的问答任务，模型输入包括问题，输出是答案的文本片段。

微调步骤：

- 模型架构：BERT-based QA模型通常是带有两个输出层的BERT模型：
  - **起始位置预测**：预测答案在文本中的起始位置。
  - **结束位置预测**：预测答案在文本中的结束位置。
- **数据格式**：需要提供配对的“问题”和“相关文本段落”作为输入，模型会输出答案的起始和结束位置。
- **训练数据**：使用像SQuAD（Stanford Question Answering Dataset）这样的问答数据集。
- **微调**：冻结部分BERT层，仅训练输出层或微调整个模型。

2、**GPT、T5等生成式模型：**

如果你希望模型生成完整的答案，而不是在给定的段落中定位答案，可以使用生成式语言模型进行微调。

- **T5（Text-to-Text Transfer Transformer）**：这类模型可以将问题和答案都视为文本生成任务。输入问题后，模型会生成一个可能的答案。
- **GPT系列模型**：这种模型擅长自然语言生成任务，尤其适合无段落输入，纯粹从问题中生成答案。

其他：**专用QA模型**

- **BART**：BART（Bidirectional and Auto-Regressive Transformers）是一个强大的生成式模型，适合进行问答任务，尤其在没有段落时生成答案的场景。

## 二、NLP

```
# 参考：https://transformers.run/c1/nlp/
```

### 1、word2vec or GloVe

上下文无关模型。Word2Vec 和 GloVe 都是两种经典的词嵌入（Word Embedding）技术，它们在自然语言处理（NLP）中用于将词语表示为连续的低维向量，使得语义相似的词在向量空间中距离较近。

- word2vec

参考：https://www.tensorflow.org/tutorials/text/word2vec?hl=zh-cn

Word2Vec 是由 Google 于 2013 年提出的模型，它有两种主要的训练方法：**CBOW（Continuous Bag of Words）** 和 **Skip-gram**。

连续词袋模型（**CBOW**）：根据周围的上下文单词预测中间单词。上下文由当前（中间）单词前后的几个单词组成。这种架构被称为词袋模型，因为上下文中的单词顺序并不重要。
连续跳字模型（ **Skip-gram**）：预测同一句子中当前单词前后一定范围内的单词。

- GloVe

参考：https://nlp.stanford.edu/projects/glove/

GloVe（Global Vectors for Word Representation）是由斯坦福大学的研究人员于 2014 年提出的模型。它采用了与 Word2Vec 不同的训练方法，基于全局统计信息。

### 2、文本生成（text generation）和 注意力机制（attention）

- **循环神经网络（RNN）文本生成**

参考：https://tensorflow.google.cn/tutorials/text/text_generation?hl=zh-cn

```python
# 代码修改：
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    inputs = tf.keras.Input(batch_shape=[batch_size, None])
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform')(x)
    outputs = tf.keras.layers.Dense(vocab_size)(x)

    model = tf.keras.Model(inputs, outputs)
    return model

# 检查点的文件名
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

model.load_weights("./training_checkpoints/ckpt_10.weights.h5")
```

循环神经网络（RNN）是一种能够处理序列数据的神经网络架构，非常适合用于处理时间序列或语言数据。RNN 在文本生成任务中广泛应用，特别是在基于序列的生成任务中，如自然语言生成、对话生成、诗歌创作等。

RNN 的变种：LSTM 和 GRU， 基本的 RNN 存在梯度消失和梯度爆炸的问题，常见的改进版本包括 LSTM（长短期记忆网络）和 GRU（门控循环单元）。

**LSTM**：LSTM 通过引入记忆单元和门控机制，有效地解决了长期依赖问题，使得模型能够记住更长时间范围内的信息。

**GRU**：GRU 是 LSTM 的简化版本，它通过减少门的数量（只有更新门和重置门）实现了更简单的结构，同时也具备处理长期依赖的能力。

- **基于注意力的神经机器翻译**

参考：https://tensorflow.google.cn/tutorials/text/transformer?hl=zh-cn

基于注意力机制的神经机器翻译（Neural Machine Translation with Attention）是现代机器翻译系统中的一个关键技术。它通过关注源句子的不同部分来改善翻译质量。这种方法解决了传统序列到序列（seq2seq）模型在长句翻译中的瓶颈问题。

**注意力机制**：在神经机器翻译中，注意力机制允许解码器在生成目标语言的每个词时，不仅依赖固定的上下文向量，还能动态选择源句子的相关部分。具体来说，注意力机制会为源句子的每个词分配一个权重，这些权重反映了每个源词对当前目标词生成的重要性。

**基本原理**

1. **编码器（Encoder）**:
   - 编码器通常是一个双向的RNN（如GRU或LSTM），它将源句子的每个词编码成一个隐状态（hidden state）。
   - 假设源句子为 `X = (x_1, x_2, ..., x_n)`，编码器生成对应的隐状态序列 `H = (h_1, h_2, ..., h_n)`。
2. **解码器（Decoder）**:
   - 解码器也是一个RNN，它根据前一时刻的隐状态和注意力机制生成的上下文向量来生成目标语言的词。
   - 在每个时间步，解码器会生成一个隐状态 `s_t` 和输出 `y_t`。
3. **注意力权重计算**:
   - 对于解码器在时间步 `t` 生成的隐状态 `s_t`，注意力机制计算它与编码器隐状态 `h_i` 之间的相关性得分。
   - 这些得分通过一个 `softmax` 函数转化为权重，称为注意力权重 `α_{t,i}`。
   - 注意力权重可以理解为“关注度”，即解码器在生成目标词 `y_t` 时，对源词 `x_i` 的关注程度。
4. **上下文向量计算**:
   - 上下文向量 `c_t` 是编码器隐状态的加权和，使用注意力权重作为权重。
   - `c_t = Σ(α_{t,i} * h_i)`
5. **生成目标词**:
   - 解码器使用当前的上下文向量 `c_t` 和当前隐状态 `s_t` 生成目标词 `y_t`

**基于注意力的神经机器翻译模型架构**

1. **编码器**:
   - 双向 LSTM 或 GRU，生成源句子的上下文表示。
2. **解码器**:
   - 单向 LSTM 或 GRU，利用注意力机制生成目标语言的词。
3. **注意力机制**:
   - 动态计算上下文向量，提供给解码器进行词生成。

### 3、Transformer

Attention Is All You Need. https://arxiv.org/abs/1706.03762  ；是一篇2017年的研究论文,提出了Transformer架构,这是一种全新的深度学习模型,完全依赖注意力机制,不再需要循环和卷积。该论文由谷歌研究员Ashish Vaswani等人撰写,对机器学习领域,尤其是自然语言处理(NLP)产生了深远的影响。

```python
参考：https://transformers.run/c1/attention/
https://github.com/datawhalechina/learn-nlp-with-transformer
https://www.cnblogs.com/mantch/p/11591937.html
```

#### 3-1、**核心思想**：

自注意力机制（self-attention）——能注意输入序列的不同位置以计算该序列的表示的能力。

-  Scaled Dot-product Attention
- Multi-Head attention
- Point wise feed forward network
- Encoder and decoder

#### 3-2、论文概读：

```


```

#### 3-3、按模型结构分类：

标准的 Transformer 模型主要由两个模块构成：

```python
（1）、输入的词语首先被转换为词向量。由于注意力机制无法捕获词语之间的位置关系，因此还通过 positional embeddings 向输入中添加位置信息；
（2）、Encoder 由一堆 encoder layers (blocks) 组成，类似于图像领域中的堆叠卷积层。同样地，在 Decoder 中也包含有堆叠的 decoder layers；
（3）、Encoder 的输出被送入到 Decoder 层中以预测概率最大的下一个词，然后当前的词语序列又被送回到 Decoder 中以继续生成下一个词，重复直至出现序列结束符 EOS 或者超过最大输出长度。
```

- **Encoder：**负责理解输入文本，为每个输入构造对应的语义表示（语义特征）；

```python
(1)第一步：tokenizer处理输入文本表示为张量格式
(2)第二步结构说明：TransformerEncoder结构 = Embeddings ==> [TransformerEncoderLayer1, TransformerEncoderLayer2, ... , TransformerEncoderLayerX]

Embeddings 结构 = （token_embeddings + position_embeddings）==> LayerNorm ==> dropout

TransformerEncoderLayer 结构 = layer normalization ==> X2 = MultiHeadAttention + X(skip connection结构) ==> feed_forward(layer normalization(X2)) + X2 (skip connection结构)
```

- **Decoder：**负责生成输出，使用 Encoder 输出的语义表示结合其他输入来生成目标序列。

```markdown
Transformer Decoder 与 Encoder 最大的不同在于 Decoder 有两个注意力子层.
  (1)**Masked multi-head self-attention layer**：确保在每个时间步生成的词语仅基于过去的输出和当前预测的词，否则 Decoder 相当于作弊了；
  (2)**Encoder-decoder attention layer**：以解码器的中间表示作为 queries，对 encoder stack 的输出 key 和 value 向量执行 Multi-head Attention。通过这种方式，Encoder-Decoder Attention Layer 就可以学习到如何关联来自两个不同序列的词语，例如两种不同的语言。 解码器可以访问每个 block 中 Encoder 的 keys 和 values。
```

根据任务的需求而单独使用：

- **纯 Encoder 模型**（例如 BERT），又称自编码 (auto-encoding) Transformer 模型；在每个阶段，注意力层都可以访问到原始输入句子中的所有词语，即具有“双向 (Bi-directional)”注意力。适用于只需要理解输入语义的任务，例如句子分类、命名实体识别（词语分类）、抽取式问答； 

- **纯 Decoder 模型**（例如 GPT），又称自回归 (auto-regressive) Transformer 模型；对于给定的词语，注意力层只能访问句子中位于它之前的词语，即只能迭代地基于已经生成的词语来逐个预测后面的词语。适用于生成式任务，例如文本生成；

- **Encoder-Decoder 模型**（例如 BART、T5），又称 Seq2Seq (sequence-to-sequence) Transformer 模型。适用于需要基于输入的生成式任务，例如翻译、摘要。

### 4、注意力机制 (Attention)

#### 4-1、Multi-Head Attention

**Attention场景常用实现方式： Scaled Dot-product Attention**：

主要思想是通过计算输入序列中各个 token 之间的相似度，决定每个 token 应该关注序列中的哪些其他 token 以及它们的权重。这一机制允许模型动态地捕捉序列中的长程依赖关系。

步骤：



```
1、输入准备（Query, Key, Value）：
	Query (Q)：表示“查询”，它是从输入中提取的，代表我们关心的 token。
	Key (K)：表示“键”，也是从输入中提取的，表示其他 token 的特征。
	Value (V)：表示“值”，同样从输入中提取，它是我们要加权平均的对象。
	通常情况下，Q、K、V 都是通过同一组输入嵌入（输入序列的表示）通过不同的线性变换生成的。
2、计算相似度得分（Attention Scores）：
	使用 Q 和 K 之间的点积来衡量 Q 与 K 的相似度：Scores=Q×K^T ; 对于每个 token，计算它与所有其他 token 之间的相似度。
3、缩放（Scaling）：
	为了防止当嵌入维度较大时，点积值变得过大（导致softmax函数梯度过小），我们将这些得分缩放：Scaled Scores= Scores/ aqrt(d_k)，d_k是 K 的维度
4、计算权重（Softmax）:
	对每个 token 的得分进行 softmax 运算，得到它对其他 token 的注意力权重;Softmax 保证了权重在 [0, 1] 之间，且对所有 token 的权重和为 1。
	Attention Weights = softmax(Scores)
5、加权求和（Weighted Sum）：
	使用这些权重对 V（Value）进行加权求和，得到最终的注意力输出：Attention Output=Attention Weights×𝑉
```

**Multi-Head Attention**:

Multi-head Attention 首先通过线性映射将 Q, K, V 序列映射到特征空间，每一组线性投影后的向量表示称为一个头 (head)，然后在每组映射后的序列上再应用 Scaled Dot-product Attention。 并行计算多个 Scaled Dot-product Attention，并在每个头的输出后进行拼接和线性变换。

通过多个注意力头，Multi-head Attention 可以在多个子空间中并行操作，每个头可能关注输入序列的不同方面。

### 5、The Feed-Forward Layer

Transformer Encoder/Decoder 中的前馈子层实际上就是两层全连接神经网络，它单独地处理序列中的每一个词向量，也被称为 position-wise feed-forward layer。常见做法是让第一层的维度是词向量大小的 4 倍，然后以 GELU 作为激活函数。

### 6、Layer Normalization

Layer Normalization 负责将一批 (batch) 输入中的每一个都标准化为均值为零且具有单位方差；

Skip Connections 则是将张量直接传递给模型的下一层而不进行处理，并将其添加到处理后的张量中。

向 Transformer Encoder/Decoder 中添加 Layer Normalization 目前共有两种做法：

    (1) Post layer normalization：Transformer 论文中使用的方式，将 Layer normalization 放在 Skip Connections 之间。 但是因为梯度可能会发散，这种做法很难训练，还需要结合学习率预热 (learning rate warm-up) 等技巧；
    (2) Pre layer normalization：目前主流的做法，将 Layer Normalization 放置于 Skip Connections 的范围内。这种做法通常训练过程会更加稳定，并且不需要任何学习率预热。

## 三、Bert

### bert 下游任务微调:SQuAD数据集

```python
https://github.com/moon-hotel/BertWithPretrained
# https://github.com/kamalkraj/BERT-SQuAD
https://github.com/surbhardwaj/BERT-QnA-Squad_2.0_Finetuned_Model
# https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset
blog: 
    https://blog.csdn.net/zcs2632008/article/details/125672908
    https://xungejiang.com/2020/06/06/BERT/ # 本文为博客 BERT Fine-Tuning Tutorial with PyTorch（https://mccormickml.com/2019/07/22/BERT-fine-tuning/） 的翻译;在本教程中，我将向你展示如何使用 BERT 与 huggingface PyTorch 库来快速高效地微调模型，以获得接近句子分类的最先进性能。
```

```
https://github.com/kamalkraj/BERT-SQuAD
```

#### 1、Bert

Bidirectional Encoder Representations from Transformers(Transformer 的双向编码器表示), https://arxiv.org/abs/1810.04805.

```
https://github.com/google-research/bert
```

```
Bert:
1、核心思想是通过Transformer架构进行双向训练，即在预训练过程中同时考虑上下文信息，从而生成更丰富的词语表征
2、BERT首先在大规模文本语料库上进行无监督的预训练（包括两个任务（MLM、NSP）：Masked Language Model和Next Sentence Prediction），然后可以在下游任务中进行有监督的微调，如分类、问答、命名实体识别等。
3、BERT基于Transformer中的Encoder部分，采用多层自注意力机制，能够有效捕捉文本中的长距离依赖关系。
```



#### 4、文本生成（text generation）和 注意力机制（attention）

- **循环神经网络（RNN）文本生成**

参考：https://tensorflow.google.cn/tutorials/text/text_generation?hl=zh-cn

```python
# 代码修改：
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    inputs = tf.keras.Input(batch_shape=[batch_size, None])
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform')(x)
    outputs = tf.keras.layers.Dense(vocab_size)(x)

    model = tf.keras.Model(inputs, outputs)
    return model

# 检查点的文件名
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

model.load_weights("./training_checkpoints/ckpt_10.weights.h5")
```

循环神经网络（RNN）是一种能够处理序列数据的神经网络架构，非常适合用于处理时间序列或语言数据。RNN 在文本生成任务中广泛应用，特别是在基于序列的生成任务中，如自然语言生成、对话生成、诗歌创作等。

RNN 的变种：LSTM 和 GRU， 基本的 RNN 存在梯度消失和梯度爆炸的问题，常见的改进版本包括 LSTM（长短期记忆网络）和 GRU（门控循环单元）。

**LSTM**：LSTM 通过引入记忆单元和门控机制，有效地解决了长期依赖问题，使得模型能够记住更长时间范围内的信息。

**GRU**：GRU 是 LSTM 的简化版本，它通过减少门的数量（只有更新门和重置门）实现了更简单的结构，同时也具备处理长期依赖的能力。

- **基于注意力的神经机器翻译**

参考：https://tensorflow.google.cn/tutorials/text/transformer?hl=zh-cn

基于注意力机制的神经机器翻译（Neural Machine Translation with Attention）是现代机器翻译系统中的一个关键技术。它通过关注源句子的不同部分来改善翻译质量。这种方法解决了传统序列到序列（seq2seq）模型在长句翻译中的瓶颈问题。

**注意力机制**：在神经机器翻译中，注意力机制允许解码器在生成目标语言的每个词时，不仅依赖固定的上下文向量，还能动态选择源句子的相关部分。具体来说，注意力机制会为源句子的每个词分配一个权重，这些权重反映了每个源词对当前目标词生成的重要性。

**基本原理**

1. **编码器（Encoder）**:
   - 编码器通常是一个双向的RNN（如GRU或LSTM），它将源句子的每个词编码成一个隐状态（hidden state）。
   - 假设源句子为 `X = (x_1, x_2, ..., x_n)`，编码器生成对应的隐状态序列 `H = (h_1, h_2, ..., h_n)`。
2. **解码器（Decoder）**:
   - 解码器也是一个RNN，它根据前一时刻的隐状态和注意力机制生成的上下文向量来生成目标语言的词。
   - 在每个时间步，解码器会生成一个隐状态 `s_t` 和输出 `y_t`。
3. **注意力权重计算**:
   - 对于解码器在时间步 `t` 生成的隐状态 `s_t`，注意力机制计算它与编码器隐状态 `h_i` 之间的相关性得分。
   - 这些得分通过一个 `softmax` 函数转化为权重，称为注意力权重 `α_{t,i}`。
   - 注意力权重可以理解为“关注度”，即解码器在生成目标词 `y_t` 时，对源词 `x_i` 的关注程度。
4. **上下文向量计算**:
   - 上下文向量 `c_t` 是编码器隐状态的加权和，使用注意力权重作为权重。
   - `c_t = Σ(α_{t,i} * h_i)`
5. **生成目标词**:
   - 解码器使用当前的上下文向量 `c_t` 和当前隐状态 `s_t` 生成目标词 `y_t`

**基于注意力的神经机器翻译模型架构**

1. **编码器**:
   - 双向 LSTM 或 GRU，生成源句子的上下文表示。
2. **解码器**:
   - 单向 LSTM 或 GRU，利用注意力机制生成目标语言的词。
3. **注意力机制**:
   - 动态计算上下文向量，提供给解码器进行词生成。

#### 5、下游任务1 ==> bert微调实例 SQuAD 2.0 Dataset： 

```python
# https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset
# 
```



#### 6、下游任务2==> PyTorch 的 BERT 微调教程: CoLA 数据集 文本分类器

```python
# https://xungejiang.com/2020/06/06/BERT/

```

