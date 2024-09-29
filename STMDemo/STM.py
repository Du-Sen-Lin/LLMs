import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import jieba
import re
import matplotlib.pyplot as plt

# 1. 加载 CSV 文件数据
data = pd.read_csv('complex_management_articles_long.csv', encoding='utf-8-sig')

# 2. 加载中文停用词
with open('chinese.txt', 'r', encoding='utf-8') as f:
    stop_words = set(f.read().splitlines())  # 改为按行读取，避免分隔符问题

# 3. 文本预处理：使用jieba分词，并移除停用词
def preprocess_text(text):
    # 去除标点符号、数字等非中文字符
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 使用 jieba 的 cut_for_search 模式进行分词
    words = jieba.cut_for_search(text)
    # 去停用词，保留长度大于1的词
    return ' '.join([word for word in words if word not in stop_words and len(word) > 1])

data['processed_text'] = data['text'].apply(preprocess_text)

# 4. 使用CountVectorizer将文本转换为词袋模型
vectorizer = CountVectorizer(max_df=0.85, min_df=2, max_features=1000)  # 添加 max_features 限制词汇表大小
X = vectorizer.fit_transform(data['processed_text'])

# 5. 使用LDA来提取主题（设定参数优化）
num_topics = 5  # 设定提取5个主题
lda_model = LatentDirichletAllocation(
    n_components=num_topics,
    max_iter=20,  # 增加迭代次数
    learning_method='batch',  # 使用 batch 方法可以提高稳定性
    random_state=42,
    n_jobs=-1  # 使用所有CPU核心
)

lda_model.fit(X)

# 6. 打印每个主题的关键词
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

n_top_words = 10
print_top_words(lda_model, vectorizer.get_feature_names_out(), n_top_words)

# 7. 可视化主题分布（保持原有的柱状图）
doc_topic_dist = lda_model.transform(X)
fig, ax = plt.subplots(figsize=(12, 8))

for i in range(num_topics):
    ax.bar(range(len(doc_topic_dist)), doc_topic_dist[:, i], label=f'Topic {i+1}', alpha=0.7)

plt.xlabel("Document Index")
plt.ylabel("Topic Proportion")
plt.title("Topic Distribution across Documents")
plt.legend()
plt.tight_layout()
plt.show()

# 8. 查看每个文档的主题分布
doc_topic_distribution = lda_model.transform(X)
df_topics = pd.DataFrame(doc_topic_distribution, columns=[f'Topic {i+1}' for i in range(num_topics)])
df_topics['article_id'] = data['article_id']  # 添加文章ID用于标识

# 保存主题分布到 CSV 文件
df_topics.to_csv('document_topic_distribution.csv', index=False, encoding='utf-8-sig')

print(df_topics)
