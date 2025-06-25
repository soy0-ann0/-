import re
import nltk
import requests
import json
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from wordcloud import WordCloud
import seaborn as sns
import torch

# 选设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 造数据（CPU 上的张量）
x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

# 移到 GPU
x = x.to(device)
y = y.to(device)

# 运算（自动用 CUDA）
z = torch.matmul(x, y)  # 矩阵乘法，GPU 加速
print("结果设备：", z.device)  # 输出 'cuda:0'（或对应 GPU 编号）
# 数据准备
news_data = [
    "Paris attacks: At least 18 dead in multiple shootings and explosions across the city",
    "Breaking: Explosions heard near Stade de France during France-Germany match",
    "ISIS claims responsibility for Paris terrorist attacks",
    "World leaders condemn Paris attacks, offer condolences to France",
    "Protests erupt in Paris following terrorist attacks",
    "Brussels airport爆炸: 34 dead in suicide bombings claimed by ISIS",
    "Turkish military convoy hit by explosion in Diyarbakir, 6 soldiers killed",
    "Golden eagle snatches baby from park in viral video (FAKE)",
    "Trump rally attacker linked to ISIS - fake news claim debunked",
    "Paris attackers planned multiple targets, the Bataclan theater hardest hit"
]

# 数据预处理
nltk.download(['stopwords', 'wordnet'])
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return lemmatized_words


processed_data = [preprocess_text(text) for text in news_data]

# 构建LDA模型
dictionary = Dictionary(processed_data)
dictionary.filter_extremes(no_below=2)
corpus = [dictionary.doc2bow(text) for text in processed_data]

lda_model = LdaModel(
    corpus=corpus,
    num_topics=3,
    id2word=dictionary,
    passes=10,
    random_state=42
)

# 可视化分析
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis_data, 'lda_vis.html')
print("\n已生成pyLDAvis交互图至lda_vis.html")


# 修复词云图中文显示
def generate_wordcloud(keywords, title):
    wordcloud = WordCloud(
        width=800,
        height=400,
        font_path='C:/Windows/Fonts/simhei.ttf',  # Windows系统默认中文字体
        background_color='white'
    ).generate(' '.join(keywords))

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis('off')
    plt.show()


# 获取LDA主题并生成词云图
lda_topics = lda_model.show_topics(formatted=False)
for topic_id, (topic_idx, words) in enumerate(lda_topics):
    print(f"主题 {topic_id + 1} 原始数据:", words)

    try:
        # 提取前8个关键词
        topic_keywords = [word for word, _ in words[:8]]
        generate_wordcloud(topic_keywords, f"主题 {topic_id + 1} 关键词云")
    except TypeError as e:
        print(f"处理主题 {topic_id + 1} 时出错: {e}")
        continue


def plot_document_topic_heatmap(lda_model, corpus, news_data):
    """绘制文档-主题概率分布热力图"""
    # 获取每篇文档的主题概率分布
    doc_topic_probs = []
    for doc in corpus:
        topic_probs = lda_model.get_document_topics(doc, minimum_probability=0)
        probs = [prob for topic_id, prob in topic_probs]
        doc_topic_probs.append(probs)

    # 转换为DataFrame（行：文档，列：主题，值：概率）
    import pandas as pd
    df = pd.DataFrame(doc_topic_probs, columns=[f"主题 {i + 1}" for i in range(lda_model.num_topics)])
    df.insert(0, "新闻文本", ["...".join(text.split()[:10]) for text in news_data])  # 显示前10个词

    # 绘制热力图
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        df.iloc[:, 1:],  # 排除新闻文本列
        annot=True,  # 显示概率值
        fmt=".2f",  # 保留两位小数
        cmap="viridis",  # 颜色映射
        cbar_kws={"label": "主题概率"},
        xticklabels=df.columns[1:],
        yticklabels=df["新闻文本"].str[:50]  # 截断长文本
    )

    plt.title("文档-主题概率分布热力图")
    plt.xlabel("主题")
    plt.ylabel("新闻文本（截断显示）")
    plt.tight_layout()
    plt.savefig("document_topic_heatmap.png")  # 保存为图片
    plt.show()


# 调用热力图函数（需在LDA模型训练后执行）
if lda_model.num_topics > 0:
    plot_document_topic_heatmap(lda_model, corpus, news_data)


# 适配Ollama API的Llama 3分析函数
def analyze_topic_with_llama3(keywords, topic_id):
    prompt = f"""
你是一位专业的新闻分析师。分析以下从新闻文本中提取的主题关键词：
主题 {topic_id + 1} 关键词: {', '.join(keywords)}

请完成以下分析：
1. 用一句话概括该主题的核心内容
2. 识别该主题涉及的主要实体（组织、人物、地点等）
3. 分析该主题的情感倾向（积极、消极、中性）
4. 预测该主题的发展趋势或影响

请按照以下格式回答：
核心内容: [一句话总结]
主要实体: [列出主要实体]
情感倾向: [情感类型]
发展趋势: [预测内容]
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "max_tokens": 500,
                "temperature": 0.3
            },
            timeout=60
        )

        response.raise_for_status()  # 检查HTTP状态码
        result = response.json()

        # 关键修改：使用'response'字段获取结果
        if 'response' in result:
            return result['response'].strip()
        else:
            print(f"API响应缺少'response'字段: {result}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"API请求错误: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"解析JSON响应失败: {e}")
        print(f"响应内容: {response.text[:200]}...")
        return None
    except Exception as e:
        print(f"未知错误: {e}")
        return None


# 获取主题关键词并使用Llama 3分析
topic_keywords = {}
for topic_id in range(lda_model.num_topics):
    words = lda_model.show_topic(topic_id, topn=8)
    keywords = [word for word, _ in words]
    topic_keywords[topic_id] = keywords

# 分析所有主题
topic_analyses = {}
for topic_id, keywords in topic_keywords.items():
    print(f"正在分析主题 {topic_id + 1}...")
    analysis = analyze_topic_with_llama3(keywords, topic_id)
    if analysis:
        topic_analyses[topic_id] = analysis
        print(f"主题 {topic_id + 1} 分析完成")

# 展示分析结果
for topic_id, analysis in topic_analyses.items():
    print(f"\n=== 主题 {topic_id + 1} 分析结果 ===")
    print(f"关键词: {', '.join(topic_keywords[topic_id])}")

    # 按段落分割并打印分析结果
    for section in analysis.split('\n\n'):
        if section.strip():
            print(section)