# Foundations-of-LLMs

```python
llm-action: https://github.com/liguodongiot/llm-action; 
LLM第14章-16章: https://transformers.run/; 
大模型基础: https://github.com/ZJU-LLMs/Foundations-of-LLMs
```

# 基础知识







# 应用

## 一、ModelScope

- 模型下载：https://modelscope.cn/docs/sdk/model-load-and-preprocess

​	下载整个模型repo（到默认cache地址）: 	` modelscope download --model 'Qwen/Qwen2-7b' `

​	下载整个模型repo到指定目录:	`  modelscope download --model 'Qwen/Qwen2-7b' --local_dir 'path/to/dir'`

​	指定下载单个文件(以'tokenizer.json'文件为例):  ` modelscope download --model 'Qwen/Qwen2-7b' tokenizer.json`

```python
from modelscope.models import Model
# 传入模型id或模型目录
model = Model.from_pretrained('some model')
```



```python
from modelscope.models import Model
# 使用模型的 ID 进行下载
model_id = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
model = Model.from_pretrained(model_id)
# 查看模型结构或模型信息
print(model)
```



- 模型推理pipeline: https://modelscope.cn/docs/sdk/pipelines 

```python
"""
pipeline构造参数
    task: 任务名称，必填
    model: 模型名称或模型实例，可选。不填时使用该任务默认模型
    preprocessor: 预处理器实例，可选。不填时使用模型配置文件中的预处理器
    device: 运行设备，可选。值为cpu, cuda, gpu, gpu:X or cuda:X，默认gpu
    device_map: 模型参数到运行设备的映射，可选，不可与device同时配置。值为auto, balance, balanced_low_0, sequential或映射dict
"""
from modelscope.pipelines import pipeline
word_segmentation = pipeline('word-segmentation')

input_str = '今天天气不错，适合出去游玩'
print(word_segmentation(input_str))

# 输出
{'output': '今天 天气 不错 ， 适合 出去 游玩'}
```



```python
from modelscope.pipelines import pipeline
from modelscope import Tasks
import re

# 输入文本
input_text = "ModelScope 是什么？"
# 执行推理
response = qa_pipeline(input_text, max_length=5000)
print(response)

# 思考过程
response_text = response['text']
cleaned_answer = re.split(r'</think>', response_text)[0]  # 思考过程
print(cleaned_answer)

# 答案
response_text = response['text']
# 去除思考部分的标记
cleaned_answer = re.split(r'</think>', response_text)[-1]  # 取最后一部分作为最终答案
print(cleaned_answer)
```



- 模型训练微调：https://modelscope.cn/docs/sdk/model-training





## 二、huggingface



```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定本地模型路径（请根据实际路径修改）
model_path = "/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 准备输入
input_text = "介绍一下ModelScope"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 生成文本
output = model.generate(input_ids, max_length=5000)
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)

```



## 三、vllm 

部署常用框架：vllm  与 SGLang

```
github: https://github.com/vllm-project/vllm
doc: https://docs.vllm.ai/en/latest/
```

VLLM是LLM推理和服务的快速易用库。

新增加TensorRT, Openvino, ModelScope 环境，可适应图像与LLM。为了增加 **VLLM** 环境用于LLM的部署推理，更新一版，原因：VLLM必须编译许多CUDA内核，该汇编引入了与其他CUDA版本和Pytorch版本的二进制不相容性，会重新安装Pytorch等版本，因此，建议安装具有新的新环境的VLLM。所有commit 一个版本，启动新的容器，安装 vllm 环境。

```python
docker cp 3d90c7d69ed8:/root/.cache/. /home/wood/Wood/common/pretrained/_.cache/

docker commit 3d90c7d69ed8 nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04-v0.0.2

# 启动并运行一个新的 Docker 容器
docker run -dit --name vllm -p 4322:22 -p 4330-4399:4330-4399 -v /home/wood/Wood/dataset:/root/dataset -v /home/wood/Wood/project:/root/project -v /home/wood/Wood/common/pretrained/_.torch:/root/.torch -v /home/wood/Wood/common/pretrained/_.cache:/root/.cache  -v /dev/shm:/dev/shm --gpus all --privileged --entrypoint /bin/bash nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04-v0.0.2

# 进入容器终端
docker exec -it vllm /bin/bash

# 启动 jupyter 服务: 只释放一个端口，统一使用xx40
nohup jupyter-notebook --no-browser --ip 0.0.0.0 --port 4340 --allow-root > jupyter.nohub.out &

# 浏览器打开
http://192.168.42.211:4340/tree?

# ssh
service ssh start
ssh -p 4322 root@192.168.42.211  # byd2025
root@192.168.42.211:4322
```

```

```





# Model

## DeepSeek-R1-Distill-Llama-8B

```python
modelscope: https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/summary
huggingface: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```



