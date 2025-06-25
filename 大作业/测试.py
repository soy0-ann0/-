import torch
import torch.nn as nn
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import pandas as pd
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
# 确保nltk资源已下载
try:
    nltk.data.find('stopwords')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# 设置transformers缓存目录
os.environ["TRANSFORMERS_CACHE"] = "./transformers_cache"


# ----------------------- 1. 定义多模态模型 ----------------------- #
class MultimodalModel(nn.Module):
    def __init__(self, topic_dim=5, hidden_dim=256):
        super().__init__()
        # 使用缓存目录加载BERT模型
        local_model_dir = "../model"  # 本地模型路径
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(
            local_model_dir,
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            output_hidden_states=False
        )
        self.text_proj = nn.Linear(768, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 1 + topic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_ids, attention_mask, emotion, topic):
        # 确保数据类型一致
        emotion = emotion.to(torch.float32)
        topic = topic.to(torch.float32)

        bert_output = self.bert(input_ids, attention_mask)[0].mean(dim=1)
        text_features = self.text_proj(bert_output)
        fused_features = torch.cat([text_features, emotion, topic], dim=1)
        logits = self.fusion(fused_features).squeeze()
        return torch.sigmoid(logits)


# ----------------------- 2. 文本预处理类 ----------------------- #
class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english')) | set(stopwords.words('french')) | {''}
        self.lemmatizer = WordNetLemmatizer()
        # 使用缓存目录加载分词器
        local_tokenizer_dir = "../model"  # 本地模型路径
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            local_tokenizer_dir,  # 本地目录路径
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            use_fast=False
        )
        self.max_length = 128

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\sÀ-ÿ]', ' ', text.lower())
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if
                  token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def encode_text(self, text):
        clean_text = self.clean_text(text)
        encoding = self.tokenizer.encode_plus(
            clean_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


# ----------------------- 3. 自定义数据集类（适配TXT文件） ----------------------- #
class TextFileDataset(Dataset):
    def __init__(self, file_path, processor, topic_num=5):
        self.processor = processor
        self.topic_num = topic_num
        self.texts = self._load_texts(file_path)
        self.emotions = self._analyze_sentiment()
        self.topics = self._predict_topics()

    def _load_texts(self, file_path):
        """从TXT文件加载文本数据，假设每行一条文本"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]

    def _analyze_sentiment(self):
        """分析文本情感（示例实现，实际应替换为真实的情感分析模型）"""
        # 这里用随机值模拟情感分析结果，实际应使用预训练的情感分析模型
        return np.random.uniform(0, 1, len(self.texts))

    def _predict_topics(self):
        """预测文本主题（示例实现，实际应替换为真实的主题模型）"""
        # 这里用随机值模拟主题分布，实际应使用预训练的主题模型
        return np.random.rand(len(self.texts), self.topic_num)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.processor.encode_text(text)

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'emotion': torch.tensor([self.emotions[idx]], dtype=torch.float32),
            'topic': torch.tensor(self.topics[idx], dtype=torch.float32),
            'text': text
        }


# ----------------------- 4. 主函数 ----------------------- #
def main():
    # 参数配置
    TEST_FILE = 'posts.txt'  # 测试数据文件路径
    MODEL_PATH = 'multimodal_news_classifier.pth'  # 模型路径
    OUTPUT_FILE = 'predictions.csv'  # 输出结果文件
    TOPIC_NUM = 5  # 主题数量
    BATCH_SIZE = 8  # 批处理大小
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 检查文件是否存在
    if not os.path.exists(TEST_FILE):
        print(f"错误: 测试文件 '{TEST_FILE}' 不存在!")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件 '{MODEL_PATH}' 不存在!")
        return

    # 确保缓存目录存在
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

    # 初始化文本处理器
    print(f"初始化文本处理器...")
    processor = TextProcessor()

    # 创建数据集和数据加载器
    print(f"加载测试数据: {TEST_FILE}")
    dataset = TextFileDataset(TEST_FILE, processor, TOPIC_NUM)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model = MultimodalModel(topic_dim=TOPIC_NUM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 预测
    print("开始预测...")
    all_predictions = []
    all_probs = []
    all_texts = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            emotion = batch['emotion'].to(DEVICE)
            topic = batch['topic'].to(DEVICE)

            outputs = model(input_ids, attention_mask, emotion, topic)
            probs = outputs.cpu().numpy()
            predictions = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_predictions.extend(predictions)
            all_texts.extend(batch['text'])

    # 转换预测结果为标签
    labels = ['real' if p == 1 else 'fake' for p in all_predictions]

    # 保存结果
    results_df = pd.DataFrame({
        'text': all_texts,
        'predicted_label': labels,
        'probability': all_probs
    })

    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"预测完成! 结果已保存到 {OUTPUT_FILE}")

    # 打印一些示例结果
    print("\n示例预测:")
    for i in range(min(5, len(all_texts))):
        print(f"\n文本: {all_texts[i][:100]}...")
        print(f"预测标签: {labels[i]}")
        print(f"预测概率: {all_probs[i]:.4f}")


if __name__ == "__main__":
    main()