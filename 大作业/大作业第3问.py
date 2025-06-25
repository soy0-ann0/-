import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
import torch

# 选设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 造数据（CPU 上的张量）
x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

# 移到 GPU
x = x.to(device)
y = y.to(device)

# ----------------------- 全局配置 ----------------------- #
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 确保nltk资源已下载
try:
    nltk.data.find('stopwords')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# ----------------------- 远程API配置 ----------------------- #
LLAMA3_API_URL = "http://localhost:11434/api/generate"  # Llama3服务地址
API_TIMEOUT = 30  # 请求超时时间（秒）


# ----------------------- 1. 数据加载与预处理 ----------------------- #
class NewsDataset(Dataset):
    def __init__(self, texts, labels, max_length=128):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

        # 直接使用镜像URL加载分词器（关键修改）
        local_tokenizer_dir = "../model"  # 本地模型路径
        self.tokenizer = BertTokenizer.from_pretrained(
            local_tokenizer_dir,  # 改为本地目录
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            use_fast=False
        )

        self.stop_words = set(stopwords.words('english')) | set(stopwords.words('french')) | {''}
        self.lemmatizer = WordNetLemmatizer()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_clean = self._clean_text(text)

        encoding = self.tokenizer.encode_plus(
            text_clean,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze().to(torch.int64),  # 明确类型
            'attention_mask': encoding['attention_mask'].squeeze().to(torch.int64),  # 明确类型
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }

    def _clean_text(self, text):
        """清洗文本（去除特殊符号、词形还原）"""
        text = re.sub(r'[^a-zA-Z\sÀ-ÿ]', ' ', text.lower())
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if
                  token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)


# ----------------------- 2. Llama3 API调用模块 ----------------------- #
class Llama3APIAnalyzer:
    def __init__(self, model_name="llama3", api_base="http://localhost:11434"):
        self.model_name = model_name
        self.api_base = api_base
        self.headers = {"Content-Type": "application/json"}

    def _call_api(self, prompt, max_tokens=512, temperature=0.7, top_p=0.9):
        """通用API调用方法（适配Ollama）"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(
                f"{self.api_base}/api/generate",
                json=payload,
                headers=self.headers,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"API调用失败: {e}")
            print(f"HTTP状态码: {response.status_code if 'response' in locals() else 'N/A'}")
            print(f"响应内容: {response.text if 'response' in locals() else 'N/A'}")
            return ""
        except Exception as e:
            print(f"处理API响应时出错: {e}")
            return ""

    def analyze_sentiment(self, text):
        """情感分析：调用远程API获取分数"""
        prompt = f"""请分析以下新闻的情感倾向（范围：-1到1，-1为负面，0为中性，1为正面）：
新闻内容：{text}
请仅返回一个保留两位小数的浮点数。"""
        # 修正参数名：从max_length改为max_tokens
        response = self._call_api(prompt, max_tokens=32, temperature=0.1)
        try:
            sentiment = float(re.search(r'[-+]?\d*\.\d+|\d+', response).group())
            return max(-1.0, min(1.0, sentiment))
        except (AttributeError, ValueError) as e:
            print(f"情感分析结果解析失败: {e}")
            print(f"响应内容: {response[:50]}...")
            return 0.0  # 默认返回中性情感

    def predict_fake_news(self, text):
        """假新闻预测：调用远程API获取结果"""
        prompt = f"""判断以下新闻是否为假新闻（仅返回0或1）：
新闻内容：{text}
请仅返回一个整数（0=假新闻，1=真实新闻）。"""
        # 修正参数名：从max_length改为max_tokens
        response = self._call_api(prompt, max_tokens=4, temperature=0.1)
        try:
            prediction = int(re.search(r'\d', response).group())
            return 1 if prediction == 1 else 0
        except (AttributeError, ValueError) as e:
            print(f"假新闻预测结果解析失败: {e}")
            print(f"响应内容: {response[:50]}...")
            return 0  # 默认返回假新闻

# ----------------------- 3. 主题分析（LDA） ----------------------- #
from gensim.corpora import Dictionary
from gensim.models import LdaModel

class TopicModeler:
    def __init__(self, num_topics=5):
        self.num_topics = num_topics
        self.dictionary = None
        self.lda_model = None

    def train_lda(self, texts):
        """训练LDA模型并提取主题特征"""
        processed_texts = self._preprocess(texts)
        self.dictionary = Dictionary(processed_texts)
        corpus = [self.dictionary.doc2bow(doc) for doc in processed_texts]
        self.lda_model = LdaModel(
            corpus=corpus,
            num_topics=self.num_topics,
            id2word=self.dictionary,
            passes=10,
            alpha='auto',
            eta='auto'
        )
        return self._get_topic_features(corpus)

    def _preprocess(self, texts):
        """文本预处理（分词、去停用词、词形还原）"""
        stop = set(stopwords.words('english') + stopwords.words('french'))
        lemmatizer = WordNetLemmatizer()
        return [
            [lemmatizer.lemmatize(token) for token in re.findall(r'\w+', text.lower()) if
             token not in stop and len(token) > 2]
            for text in texts
        ]

    def _get_topic_features(self, corpus):
        """提取每个文档的主题概率分布"""
        topic_features = []
        for doc_bow in corpus:
            topic_dist = [0.0] * self.num_topics
            for topic_id, prob in self.lda_model.get_document_topics(doc_bow):
                topic_dist[topic_id] = prob
            topic_features.append(topic_dist)
        return np.array(topic_features)


# ----------------------- 4. 多模态融合模型 ----------------------- #
class MultimodalModel(nn.Module):
    def __init__(self, topic_dim=5, hidden_dim=256):
        super().__init__()
        local_model_dir = "../model"  # 本地模型路径
        self.bert = BertModel.from_pretrained(
            local_model_dir,
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            output_hidden_states=False
        )
        self.text_proj = nn.Linear(768, hidden_dim)

        # 多模态融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 1 + topic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_ids, attention_mask, emotion, topic):
        # 强制转换为float32
        emotion = emotion.to(torch.float32)
        topic = topic.to(torch.float32)

        bert_output = self.bert(input_ids, attention_mask)[0].mean(dim=1)
        text_features = self.text_proj(bert_output)
        fused_features = torch.cat([text_features, emotion, topic], dim=1)
        logits = self.fusion(fused_features).squeeze()
        return torch.sigmoid(logits)


# ----------------------- 5. 主流程整合 ----------------------- #
def main():
    # 配置参数
    DATA_PATH = 'true_fake_labels.txt'  # 数据文件路径
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TOPIC_NUM = 5  # LDA主题数
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    # 读取数据
    print(f"加载数据: {DATA_PATH}")
    try:
        data = pd.read_csv(DATA_PATH, sep='\t',
                           names=['post_id', 'post_text', 'user_id', 'username', 'image_id', 'timestamp', 'label'])
        print(f"数据加载成功，共{len(data)}条记录")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return


    random_seed = 42
    sampled_data = data.sample(n=1000, random_state=random_seed)
    texts = sampled_data['post_text'].tolist()
    true_labels = sampled_data['label'].map({'fake': 0, 'real': 1}).tolist()

    # 创建基础数据集
    dataset = NewsDataset(texts, true_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化Llama3 API分析器
    print("初始化Llama3 API分析器...")
    llama3 = Llama3APIAnalyzer()

    # 提取情感特征（通过API调用）
    print("提取情感特征...")
    emotions = []
    for batch in tqdm(dataloader):
        batch_emotions = [llama3.analyze_sentiment(text) for text in batch['text']]
        emotions.extend(batch_emotions)

    # 训练主题模型（LDA）
    print("训练主题模型...")
    topic_modeler = TopicModeler(num_topics=TOPIC_NUM)
    topic_features = topic_modeler.train_lda(texts)

    # 创建多模态数据集
    class MultimodalDataset(Dataset):
        def __init__(self, dataset, emotions, topic_features):
            self.dataset = dataset
            self.emotions = emotions
            self.topic_features = topic_features

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            # 明确指定float32
            item['emotion'] = torch.tensor([self.emotions[idx]], dtype=torch.float32)
            item['topic'] = torch.tensor(self.topic_features[idx], dtype=torch.float32)
            return item

    multimodal_dataset = MultimodalDataset(dataset, emotions, topic_features)
    multimodal_dataloader = DataLoader(multimodal_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化并训练多模态模型
    print("训练多模态模型...")
    model = MultimodalModel(topic_dim=TOPIC_NUM).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(multimodal_dataloader):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            emotion = batch['emotion'].to(DEVICE)
            topic = batch['topic'].to(DEVICE)
            labels = batch['label'].to(DEVICE).float()

            # 前向传播
            outputs = model(input_ids, attention_mask, emotion, topic)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(multimodal_dataloader):.4f}")

    # 评估模型
    print("评估模型...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in multimodal_dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            emotion = batch['emotion'].to(DEVICE)
            topic = batch['topic'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids, attention_mask, emotion, topic)
            preds = (outputs > 0.5).long()

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f"Accuracy: {correct / total:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'multimodal_news_classifier.pth')
    print("模型已保存为: multimodal_news_classifier.pth")

    # 示例预测
    if len(texts) > 0:
        sample_idx = 0
        sample_text = texts[sample_idx]
        sample_input = dataset[sample_idx]

        model.eval()
        with torch.no_grad():
            input_ids = sample_input['input_ids'].unsqueeze(0).to(DEVICE)
            attention_mask = sample_input['attention_mask'].unsqueeze(0).to(DEVICE)
            emotion = torch.tensor([[emotions[sample_idx]]]).to(DEVICE)
            topic = torch.tensor([topic_features[sample_idx]]).to(DEVICE)

            prediction = model(input_ids, attention_mask, emotion, topic).item()
            true_label = true_labels[sample_idx]

            print("\n示例预测:")
            print(f"新闻内容: {sample_text[:100]}...")
            print(f"真实标签: {'真实' if true_label == 1 else '虚假'}")
            print(f"预测概率: {prediction:.4f} ({'真实' if prediction > 0.5 else '虚假'})")


if __name__ == "__main__":
    main()