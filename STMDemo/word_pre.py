from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
import jieba

# 读取Word文档
def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ' '.join(full_text)


# def preprocess(text):
#     # 中文分词
#     tokens = jieba.cut(text)
#     # 过滤停用词和单字
#     filtered_tokens = [word for word in tokens if word != ' ' and len(word) > 1]
#     return filtered_tokens

# 中文分词并过滤停用词
def preprocess(text):
    stop_words = set(stopwords.words('chinese'))
    tokens = jieba.cut(text)
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return filtered_tokens


# 读取Word文档
text = read_docx('qingshan.docx')
print(text)

# 文本预处理:使用jieba进行中文分词，并进行简单的预处理。
processed_text = preprocess(text)
print(processed_text)

# 构建文档-词矩阵
# 使用gensim构建词典和文档-词矩阵。
from gensim import corpora

dictionary = corpora.Dictionary([processed_text])
corpus = [dictionary.doc2bow(processed_text)]
print(f"--------------- corpus ----------------")
print(f"{corpus}")

# 训练STM模型
# 使用gensim中的LdaModel进行模型训练。
from gensim.models import LdaModel

lda = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# 主题解释
# 打印主题的关键词。
topics = lda.print_topics(num_words=4)
print(f"--------------- topics ----------------")
for topic in topics:
    print(topic)

# 结果评估
# 评估模型通常涉及到模型的困惑度（Perplexity）。
perplexity = lda.log_perplexity(corpus)
print(f'Perplexity: {perplexity}')