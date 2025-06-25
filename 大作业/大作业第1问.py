import pandas as pd
import requests
import json
from sklearn.metrics import accuracy_score, classification_report

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


# 读取数据
data = pd.read_csv('true_fake_labels.txt', sep='\t',
                   names=['post_id', 'post_text', 'user_id', 'username', 'image_id', 'timestamp', 'label'])

random_seed = 42
sampled_data = data.sample(n=1000, random_state=random_seed)
texts = sampled_data['post_text'].tolist()
true_labels = sampled_data['label'].map({'fake': 0, 'real': 1}).tolist()

def create_judge_prompt(news_text):
    return f"""判断以下新闻是真新闻（输出1）还是假新闻（输出0），不要包含其他任何内容：
新闻内容：{news_text}
必须只返回一个字符：0或1"""

def create_sentiment_prompt(news_text):
    return f"""分析以下新闻的语义情感倾向，仅返回数字：
- 强烈积极：1
- 轻微积极：0.5
- 中性：0
- 轻微消极：-0.5
- 强烈消极：-1

新闻内容：{news_text}
必须只返回一个数字，如：1, 0.5, 0, -0.5, -1"""

def create_judge_with_sentiment_prompt(news_text, sentiment):
    sentiment_desc = "强烈积极" if sentiment == 1 else \
                    "轻微积极" if sentiment == 0.5 else \
                    "中性" if sentiment == 0 else \
                    "轻微消极" if sentiment == -0.5 else "强烈消极"
    return f"""判断以下新闻的真实性（0为假新闻，1为真新闻），不要受情感倾向影响：

新闻内容：{news_text}
情感倾向：{sentiment_desc}

只返回一个字符：0或1"""

def parse_response(response_text, task_type="judge"):
    """解析流式JSON响应，提取完整预测结果"""
    print(f"原始响应：{response_text[:200]}")

    # 按行分割响应（每行是一个JSON对象）
    lines = response_text.strip().split('\n')
    full_response = ""

    for line in lines:
        if line.strip():
            try:
                data = json.loads(line)
                if 'response' in data:
                    full_response += data['response']
            except json.JSONDecodeError as e:
                print(f"解析行失败：{line[:50]}... 错误：{e}")
                continue

    # 针对不同任务类型解析不同的响应格式
    if task_type == "sentiment":
        # 解析情感值
        if "1" in full_response and "-" not in full_response:
            return 1.0
        elif "0.5" in full_response and "-" not in full_response:
            return 0.5
        elif "0" in full_response:
            return 0.0
        elif "0.5" in full_response and "-" in full_response:
            return -0.5
        elif "-1" in full_response:
            return -1.0
        else:
            print(f"情感分析未找到有效值，内容：{full_response}")
            return None
    else:
        # 解析判断结果
        if "0" in full_response:
            return 0
        elif "1" in full_response:
            return 1
        else:
            print(f"判断未找到有效值，内容：{full_response}")
            return None

def predict_with_llama3(texts, model_name="llama3", prompt_func=None, task_type="judge"):
    predictions = []
    for idx, text in enumerate(texts):
        truncated_text = text[:4000]
        if prompt_func:
            prompt = prompt_func(truncated_text)
        else:
            prompt = create_judge_prompt(truncated_text)
        pred = None
        retries = 3

        while retries > 0:
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "max_tokens": 10,
                        "stop": ["\n", "0", "1", "0.5", "-0.5", "-1"],
                        "temperature": 0.0
                    },
                    timeout=30
                )
                response.raise_for_status()
                pred = parse_response(response.text, task_type)
                break
            except Exception as e:
                print(f"样本 {idx} 尝试 {3 - retries + 1}/3 失败：{e}")
                retries -= 1

        predictions.append(pred)

        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{len(texts)}")

    return predictions

def calculate_accuracies(true_labels, predictions):
    valid_indices = [i for i, p in enumerate(predictions) if p in [0, 1]]
    if not valid_indices:
        print("警告：未获得任何有效预测！请检查Ollama服务或API响应格式。")
        return None, None, None
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_labels = [true_labels[i] for i in valid_indices]

    accuracy = accuracy_score(valid_labels, valid_predictions)

    fake_indices = [i for i, label in enumerate(valid_labels) if label == 0]
    true_indices = [i for i, label in enumerate(valid_labels) if label == 1]

    if fake_indices:
        accuracy_fake = accuracy_score([valid_labels[i] for i in fake_indices], [valid_predictions[i] for i in fake_indices])
    else:
        accuracy_fake = None

    if true_indices:
        accuracy_true = accuracy_score([valid_labels[i] for i in true_indices], [valid_predictions[i] for i in true_indices])
    else:
        accuracy_true = None

    return accuracy, accuracy_fake, accuracy_true

# 任务1：仅判断真假新闻
predictions_task1 = predict_with_llama3(texts)
accuracy_task1, accuracy_fake_task1, accuracy_true_task1 = calculate_accuracies(true_labels, predictions_task1)

# 任务2：分析语义情感
sentiments = predict_with_llama3(texts, prompt_func=create_sentiment_prompt, task_type="sentiment")

# 任务3：结合情感分析判断真假新闻
predictions_task3 = []
for text, sentiment in zip(texts, sentiments):
    if sentiment is not None:
        prompt = create_judge_with_sentiment_prompt(text, sentiment)
        pred = predict_with_llama3([text], prompt_func=lambda x: prompt)[0]
        predictions_task3.append(pred)
    else:
        predictions_task3.append(None)

# 移除无效预测
valid_predictions_task3 = []
valid_labels_task3 = []
for pred, label in zip(predictions_task3, true_labels):
    if pred in [0, 1]:
        valid_predictions_task3.append(pred)
        valid_labels_task3.append(label)

accuracy_task3, accuracy_fake_task3, accuracy_true_task3 = calculate_accuracies(true_labels, predictions_task3)

# 任务4：分析准确率是否有提升
print("\n=== 任务1结果 ===")
print(f"总准确率：{accuracy_task1 if accuracy_task1 is not None else 'N/A':.4f}")
print(f"假新闻准确率：{accuracy_fake_task1 if accuracy_fake_task1 is not None else 'N/A':.4f}")
print(f"真新闻准确率：{accuracy_true_task1 if accuracy_true_task1 is not None else 'N/A':.4f}")

print("\n=== 任务3结果 ===")
print(f"总准确率：{accuracy_task3 if accuracy_task3 is not None else 'N/A':.4f}")
print(f"假新闻准确率：{accuracy_fake_task3 if accuracy_fake_task3 is not None else 'N/A':.4f}")
print(f"真新闻准确率：{accuracy_true_task3 if accuracy_true_task3 is not None else 'N/A':.4f}")

if accuracy_task1 is not None and accuracy_task3 is not None:
    if accuracy_task3 > accuracy_task1:
        print("\n结合情感分析后，总准确率有提升！")
    elif accuracy_task3 < accuracy_task1:
        print("\n结合情感分析后，总准确率下降了！")
    else:
        print("\n结合情感分析后，总准确率没有变化。")

if accuracy_fake_task1 is not None and accuracy_fake_task3 is not None:
    if accuracy_fake_task3 > accuracy_fake_task1:
        print("结合情感分析后，假新闻准确率有提升！")
    elif accuracy_fake_task3 < accuracy_fake_task1:
        print("结合情感分析后，假新闻准确率下降了！")
    else:
        print("结合情感分析后，假新闻准确率没有变化。")

if accuracy_true_task1 is not None and accuracy_true_task3 is not None:
    if accuracy_true_task3 > accuracy_true_task1:
        print("结合情感分析后，真新闻准确率有提升！")
    elif accuracy_true_task3 < accuracy_true_task1:
        print("结合情感分析后，真新闻准确率下降了！")
    else:
        print("结合情感分析后，真新闻准确率没有变化。")

# 分析情感分布与预测结果的关系
if sentiments and predictions_task3:
    print("\n=== 情感分布与预测结果分析 ===")
    sentiment_counts = {1.0: 0, 0.5: 0, 0: 0, -0.5: 0, -1.0: 0}
    sentiment_predictions = {1.0: [], 0.5: [], 0: [], -0.5: [], -1.0: []}
    sentiment_correct = {1.0: 0, 0.5: 0, 0: 0, -0.5: 0, -1.0: 0}

    for i, (sentiment, pred, label) in enumerate(zip(sentiments, predictions_task3, true_labels)):
        if sentiment is not None and pred in [0, 1]:
            sentiment_counts[sentiment] += 1
            sentiment_predictions[sentiment].append(pred)
            if pred == label:
                sentiment_correct[sentiment] += 1

    for sentiment, count in sentiment_counts.items():
        if count > 0:
            accuracy = sentiment_correct[sentiment] / count
            sentiment_desc = "强烈积极" if sentiment == 1 else \
                            "轻微积极" if sentiment == 0.5 else \
                            "中性" if sentiment == 0 else \
                            "轻微消极" if sentiment == -0.5 else "强烈消极"
            print(f"{sentiment_desc}新闻: {count}条, 准确率: {accuracy:.4f}")