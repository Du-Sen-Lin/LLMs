# -*- coding: utf-8 -*-
import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 读取 CSV 文件
df = pd.read_csv('complex_management_articles_long.csv')

# 定义种子词（关键词）
seed_keywords = {
    "主题1": ["全球化", "市场", "跨国", "并购", "技术"],
    "主题2": ["技术", "创新", "研发", "数字化", "信息"],
    "主题3": ["供应链", "管理", "风险", "优化", "透明度"],
    "主题4": ["财务", "透明度", "风险", "合规", "审计"],
    "主题5": ["社会责任", "可持续性", "环保", "公益", "能源"]
}

# 去重的种子词列表
all_seed_words = list(set(word for words in seed_keywords.values() for word in words))

# 文本预处理函数
def preprocess_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  # 移除非中文字符
    words = jieba.cut(text)
    return ' '.join(words)

# 处理后的文本
processed_texts = [preprocess_text(text) for text in df['text']]

# 构建CountVectorizer，同时扩展自定义种子词的词汇表
vectorizer = CountVectorizer(vocabulary=all_seed_words)
X = vectorizer.fit_transform(processed_texts)

# 定义主题数量
num_topics = len(seed_keywords)

# 训练LDA模型
lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, random_state=42)

# 使用包含种子词的向量化文本数据训练LDA模型
lda.fit(X)

# 输出每个主题的关键词
def print_top_words(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

print_top_words(lda, vectorizer.get_feature_names_out())

# 获取每个文档的主题分布
doc_topic_dist = lda.transform(X)
df_topics = pd.DataFrame(doc_topic_dist, columns=[f'Topic {i+1}' for i in range(num_topics)])
df_with_topics = pd.concat([df, df_topics], axis=1)

# 保存结果
df_with_topics.to_csv('articles_with_topics.csv', index=False, encoding='utf-8-sig')

print("处理完成，结果已保存到 'articles_with_topics.csv'")
