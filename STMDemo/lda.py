from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
import jieba
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis


def read_docx(file_path):
    """
    读取文档
    :param file_path: 给定路径读取 .docx 文档
    :return: 返回其中的所有文本
    """
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ' '.join(full_text)


def preprocess(text):
    """
    文本预处理：分词和去除停用词
    :param text:
    :return:
    """
    stop_words = set(stopwords.words('chinese'))
    tokens = jieba.cut(text)
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return filtered_tokens


def build_corpus(texts):
    """
    构建词典和语料库
    :param texts:
    :return:
    """
    dictionary = corpora.Dictionary(texts)  # 构建词典
    corpus = [dictionary.doc2bow(text) for text in texts]  # 转换为词袋模型格式
    return dictionary, corpus


def train_lda_model(corpus, dictionary, num_topics):
    """
    训练LDA模型
    :param corpus:
    :param dictionary:
    :param num_topics: 提取主题数量
    :return:
    """
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=20)
    return lda_model


def compute_coherence(lda_model, texts, dictionary, corpus):
    """
    主题一致性得分
    :param lda_model:
    :param texts:
    :param dictionary:
    :param corpus:
    :return:
    """
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, corpus=corpus, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    return coherence_lda


def visualize_topics(lda_model, corpus, dictionary):
    """
    主题提取与可视化
    :param lda_model:
    :param corpus:
    :param dictionary:
    :return:
    """
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    # pyLDAvis.show(vis_data)
    pyLDAvis.save_html(vis_data, 'lda_visualization.html')  # 保存为 HTML 文件
    print("Visualization saved to lda_visualization.html")


if __name__ == '__main__':
    # 1. 读取文档
    texts = [read_docx('qingshan.docx'), read_docx('qingshan.docx')]
    print(texts)

    # 2. 文本预处理
    processed_texts  = [preprocess(text) for text in texts]
    print(processed_texts)

    # 3. 构建语料库
    dictionary, corpus = build_corpus(processed_texts)
    print("Dictionary:", dictionary)
    print("Corpus:", corpus)

    # 4. 训练LDA模型
    lda_model = train_lda_model(corpus, dictionary, num_topics=10)  # 假设提取5个主题
    print("LDA Model:", lda_model.print_topics())
    # 打印主题的关键词。
    topics = lda_model.print_topics(num_words=20)
    print(f"--------------- topics ----------------")
    for topic in topics:
        print(topic)

    # 5. 主题一致性得分
    # coherence_score = compute_coherence(lda_model, processed_texts, dictionary, corpus)
    # print("Coherence Score:", coherence_score)
    # 使用困惑度（Perplexity）评估模型
    perplexity = lda_model.log_perplexity(corpus)
    print("Model Perplexity:", perplexity)

    # 6. 主题可视化
    visualize_topics(lda_model, corpus, dictionary)
