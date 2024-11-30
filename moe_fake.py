import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from sklearn.metrics import classification_report, confusion_matrix
import json
import numpy as np
import os
from datetime import datetime


def calculate_metrics(y_true, y_pred):
    """
    计算精确率、召回率和F1分数
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    return [
        precision_score(y_true, y_pred, average='binary'),
        recall_score(y_true, y_pred, average='binary'),
        f1_score(y_true, y_pred, average='binary')
    ]
def load_fake_news_classifier(model_path):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path,trust_remote_code=True,device_map="cuda:1")
    return  model,tokenizer
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                torch_dtype=torch.bfloat16, 
                                                device_map="cuda:0",
                                                trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return  model,tokenizer
def predict_fake( model,tokenizer,title,text):
    input_str = "<title>" + title + "<content>" +  text + "<end>"
    input_ids = tokenizer.encode_plus(input_str, max_length=512, padding="max_length", truncation=True, return_tensors="pt")

    with torch.no_grad():
        output = model(input_ids["input_ids"].to(model.device), attention_mask=input_ids["attention_mask"].to(model.device))
    return dict(zip(["Fake","Real"], [x.item() for x in list(torch.nn.Softmax()(output.logits)[0])] ))
    
def generate_news(model, tokenizer, prompt, max_length=4096):
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    # Generate
    outputs = model.generate(
        # inputs["input_ids"],
        **inputs.to(model.device),
        max_length=max_length,
        num_beams=5,
        temperature=0.8,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
    )
    # Decode
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
def create_fake_title_prompt(real_title):
    prompt = f"""Task: Generate a misleading and false news title.
Reference Title: {real_title}

Requirements:
1. Create a sensational and misleading title
2. Keep the topic similar but add false information
3. Make it attention-grabbing and controversial
4. Keep it concise and newspaper-style

Generated Title:"""
    return prompt

def create_fake_content_prompt(real_text, fake_title):
    prompt = f"""Task: Generate fake news content for the following title.
Title: {fake_title}

Reference Content: {real_text}

Requirements:
1. Write a detailed fake news article supporting the misleading title
2. Include fabricated quotes, statistics, or events
3. Maintain a professional news writing style
4. Mix truth with fiction to make it more believable
5. Keep the length and structure similar to real news

Generated Content:"""
    return prompt

def generate_and_parse_fake_news(model, tokenizer, real_title, real_text):
    """
    分两步生成假新闻：先生成标题，再基于标题生成内容
    """
    try:
        # 1. 生成假新闻标题
        title_prompt = create_fake_title_prompt(real_title)
        generated_title_full = generate_news(model, tokenizer, title_prompt, max_length=100)
        
        # 解析生成的标题（去除prompt部分）
        generated_title = generated_title_full.split("Generated Title:")[-1].strip()
        
        # 2. 基于生成的标题生成假新闻内容
        content_prompt = create_fake_content_prompt(real_text, generated_title)
        generated_content_full = generate_news(model, tokenizer, content_prompt, max_length=2000)
        
        # 解析生成的内容（去除prompt部分）
        generated_content = generated_content_full.split("Generated Content:")[-1].strip()
        res = {
            'title': generated_title,
            'content': generated_content,
            'original_title': real_title,
            'original_content': real_text
        }
        print(res)
        return res
    except Exception as e:
        print(f"Error in generation: {str(e)}")
        return None
def save_results(results, base_path="/home/ma-user/work/yangwenhan/MOE/res"):
    """
    保存生成结果和评估指标
    """
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 确保目录存在
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # 1. 保存生成的假新闻示例
    examples_path = os.path.join(base_path, f'generated_fake_news_{timestamp}.json')
    with open(examples_path, 'w', encoding='utf-8') as f:
        json.dump(results['generated_examples'], f, ensure_ascii=False, indent=2)
    
    # 2. 保存评估指标
    metrics_df = pd.DataFrame({
        'metric': [
            'original_accuracy',
            'generated_accuracy',
            'fake_news_generation_success_rate',
            'original_precision',
            'original_recall',
            'original_f1',
            'generated_precision',
            'generated_recall',
            'generated_f1'
        ],
        'value': [
            np.mean(np.array(results['original_predictions']) == np.array(results['original_labels'])),
            np.mean(np.array(results['generated_predictions']) == np.array(results['generated_labels'])),
            np.mean([pred == 1 for pred in results['generated_predictions']]),
            *calculate_metrics(results['original_labels'], results['original_predictions']),
            *calculate_metrics(results['generated_labels'], results['generated_predictions'])
        ]
    })
    
    metrics_path = os.path.join(base_path, f'evaluation_metrics_{timestamp}.csv')
    metrics_df.to_csv(metrics_path, index=False)
    
    # 3. 保存详细的预测结果
    predictions_df = pd.DataFrame({
        'original_title': [ex['original_title'] for ex in results['generated_examples']],
        'generated_title': [ex['title'] for ex in results['generated_examples']],
        'original_prediction': results['original_predictions'],
        'generated_prediction': results['generated_predictions'],
        'original_label': results['original_labels'],
        'generated_label': results['generated_labels']
    })
    
    predictions_path = os.path.join(base_path, f'predictions_{timestamp}.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    return {
        'examples_path': examples_path,
        'metrics_path': metrics_path,
        'predictions_path': predictions_path
    }

def calculate_metrics(y_true, y_pred):
    """
    计算精确率、召回率和F1分数
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    return [
        precision_score(y_true, y_pred, average='binary'),
        recall_score(y_true, y_pred, average='binary'),
        f1_score(y_true, y_pred, average='binary')
    ]
def evaluate_generated_fake_news(df, model, tokenizer, fake_news_classifier, fake_news_tokenizer):
    """
    评估生成的假新闻
    """
    results = {
        'original_predictions': [],
        'generated_predictions': [],
        'original_labels': [],
        'generated_labels': [],
        'generated_examples': [],
        'prediction_scores': {
            'original': [],
            'generated': []
        }
    }
    sample_df = df.sample(n=min(num_samples, len(df)))
    
    for idx, row in sample_df.iterrows():
        # 评估原始新闻
        original_pred = predict_fake(fake_news_classifier, fake_news_tokenizer, 
                                  row['title'], row['text'])
        results['original_predictions'].append(1 if original_pred['Fake'] > 0.5 else 0)
        results['prediction_scores']['original'].append(original_pred)
        results['original_labels'].append(row['label'])
        
        # 生成并评估假新闻
        generated_news = generate_and_parse_fake_news(model, tokenizer, 
                                                    row['title'], row['text'])
        
        if generated_news:
            # 评估生成的假新闻
            generated_pred = predict_fake(fake_news_classifier, fake_news_tokenizer,
                                       generated_news['title'], 
                                       generated_news['content'])
            
            results['generated_predictions'].append(1 if generated_pred['Fake'] > 0.5 else 0)
            results['prediction_scores']['generated'].append(generated_pred)
            results['generated_labels'].append(1)  # 生成的都是假新闻
            results['generated_examples'].append(generated_news)
    return results


def print_evaluation_results(results):
    # 评估原始数据集的分类性能
    print("\nClassifier Performance on Original Dataset:")
    print(classification_report(results['original_labels'], 
                              results['original_predictions']))
    
    # 评估生成的假新闻的分类性能
    print("\nClassifier Performance on Generated Fake News:")
    print(classification_report(results['generated_labels'], 
                              results['generated_predictions']))
    
    # 计算生成假新闻的成功率（被正确分类为假新闻的比例）
    success_rate = np.mean([pred == 1 for pred in results['generated_predictions']])
    print(f"\nFake News Generation Success Rate: {success_rate:.2%}")

def main():
    # 加载数据和模型
    data_file="/home/ma-user/work/yangwenhan/MOE/data/evaluation.csv"
    save_path = "/home/ma-user/work/yangwenhan/MOE/res"
    model_name = "/home/ma-user/work/yangwenhan/MOE/hub"
    classifier_path="/home/ma-user/work/yangwenhan/huggingface/roberta-fake"
    df = pd.read_csv(data_file, sep=';')
    
    # 加载MoE模型
    
    model,tokenizer =load_model(model_name)
    
    # 加载假新闻分类器
    fake_news_classifier, fake_news_tokenizer = load_fake_news_classifier(classifier_path)
    
    # 进行评估
    results = evaluate_generated_fake_news(df, model, tokenizer, 
                                    fake_news_classifier, fake_news_tokenizer)
    # 保存结果
    saved_paths = save_results(results, save_path)
    
    # 打印保存位置
    print("\n=== Results saved to ===")
    print(f"Generated examples: {saved_paths['examples_path']}")
    print(f"Evaluation metrics: {saved_paths['metrics_path']}")
    print(f"Detailed predictions: {saved_paths['predictions_path']}")
    
    # 打印主要指标
    metrics_df = pd.read_csv(saved_paths['metrics_path'])
    print("\n=== Key Metrics ===")
    print(metrics_df)

if __name__ == "__main__":
    num_samples=10
    main()
    
