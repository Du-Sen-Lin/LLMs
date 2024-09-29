import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
import jieba

# 确保nltk的数据目录正确设置
nltk.data.path.append('C:\\nltk_data\\stopwords')

# 加载停用词
stop_words = stopwords.words('chinese')
print(stop_words)  # 打印停用词列表，以验证数据包是否正确加载