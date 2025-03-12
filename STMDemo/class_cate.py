import jieba
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

# 1. 假设有更多标注数据
data = [
    ("市场全球化对公司策略的重要性", "主题1"),
    ("技术创新如何提升企业竞争力", "主题2"),
    ("供应链管理的优化与风险控制", "主题3"),
    ("公司进入全球市场的战略方法", "主题1"),
    ("技术开发在企业中的应用", "主题2"),
    ("如何降低供应链中的风险", "主题3"),
    ("跨国公司如何应对全球市场挑战", "主题1"),
    ("创新是企业发展的核心驱动力", "主题2"),
    ("供应链管理对企业的财务影响", "主题3"),
    ("市场变化对跨国公司战略的影响", "主题1"),
    ("新兴技术如何重塑行业竞争格局", "主题2"),
    ("供应链中的成本控制与优化策略", "主题3"),
    ("公司在全球市场中的定位策略", "主题1"),
    ("技术驱动的产品开发与创新", "主题2"),
    ("全球供应链的透明度和效率提升", "主题3"),
    ("国际市场进入的机会与挑战", "主题1"),
    ("大数据在技术创新中的应用", "主题2"),
    ("如何应对全球供应链中的潜在风险", "主题3"),
    ("全球化对本土化战略的影响", "主题1"),
    ("技术创新如何推动企业数字化转型", "主题2"),
    ("供应链管理如何帮助企业提升竞争力", "主题3"),
    ("全球战略的本土化实施路径", "主题1"),
    ("人工智能技术在创新中的作用", "主题2"),
    ("供应链中的物流管理优化", "主题3"),
    ("跨国公司应对不同市场的文化差异", "主题1"),
    ("区块链技术在供应链中的应用", "主题2"),
    ("供应链弹性和风险应对策略", "主题3"),
    ("全球市场战略中的品牌定位", "主题1"),
    ("智能制造技术如何改变供应链管理", "主题2"),
    ("供应链中断的应急管理措施", "主题3"),
    ("公司应对全球化带来的市场竞争", "主题1"),
    ("数字化技术如何帮助企业创新", "主题2"),
    ("供应链管理中的信息共享与协作", "主题3")
]
texts, labels = zip(*data)


# 2. 文本预处理
def preprocess_text(text):
    # 去除标点符号和数字，只保留中文字符
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 使用jieba分词
    words = jieba.cut(text)
    return ' '.join(words)


# 处理后的文本
processed_texts = [preprocess_text(text) for text in texts]

# 3. 标签向量化
label_dict = {"主题1": 0, "主题2": 1, "主题3": 2}
y = [label_dict[label] for label in labels]

# 4. 文本向量化（使用TfidfVectorizer）
vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, ngram_range=(1, 2))  # ngram_range=(1,2)捕捉单词和双词短语
X = vectorizer.fit_transform(processed_texts)

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 训练Naive Bayes分类模型
clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)

# 7. 训练SVM分类器
clf_svm = LinearSVC()
clf_svm.fit(X_train, y_train)

# 8. 评估模型准确性
y_pred_nb = clf_nb.predict(X_test)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb)}")

y_pred_svm = clf_svm.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}")

# 9. 使用交叉验证评估模型
cross_val_scores = cross_val_score(clf_nb, X, y, cv=5)
print(f"Naive Bayes Cross-Validation Accuracy: {cross_val_scores.mean()}")

cross_val_scores_svm = cross_val_score(clf_svm, X, y, cv=5)
print(f"SVM Cross-Validation Accuracy: {cross_val_scores_svm.mean()}")


# 10. 新数据推理函数
def predict_new_data(new_text, model):
    # 对新文本进行预处理和向量化
    new_text_processed = preprocess_text(new_text)
    new_text_vectorized = vectorizer.transform([new_text_processed])

    # 使用指定模型进行预测
    prediction = model.predict(new_text_vectorized)

    # 返回预测的主题类别
    for key, value in label_dict.items():
        if value == prediction[0]:
            return key


# 示例推理
new_text = "企业如何通过技术创新在全球市场中取得竞争优势"
predicted_label_nb = predict_new_data(new_text, clf_nb)
predicted_label_svm = predict_new_data(new_text, clf_svm)
print(f"Naive Bayes Prediction for new text: {predicted_label_nb}")
print(f"SVM Prediction for new text: {predicted_label_svm}")
