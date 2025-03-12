import pandas as pd

# 创建数据
data = {
    'article_id': [1, 2, 3, 4, 5],
    'year': [2000, 2001, 2005, 2010, 2020],
    'text': [
        "公司提高了生产效率，优化了供应链管理。",
        "随着市场竞争加剧，创新成为了企业发展的核心动力。",
        "企业开始关注可持续发展与环保议题。",
        "生产自动化成为了许多制造企业的焦点。",
        "数字化转型和绿色技术在企业中得到了广泛应用。"
    ]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 保存为 CSV 文件
df.to_csv('management_articles.csv', index=False, encoding='utf-8-sig')  # 使用 utf-8-sig 以确保中文编码
