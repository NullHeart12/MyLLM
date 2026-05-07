import json
from tqdm import tqdm

# pretrain_data 为运行download_dataset.sh时，下载的pretrain_data本地路径
read_pretrain_data = './dataset/mobvoi_seq_monkey_general_open_corpus.jsonl'
output_pretrain_data = 'seq_monkey.jsonl'

# sft_data 为运行download_dataset.sh时，下载的sft_data本地路径
read_sft_data = './dataset/BelleGroup/train_3.5M_CN.json'
output_sft_data = 'BelleGroup_sft.jsonl'

# 1 处理预训练数据
def split_text(text, chunk_size=512):
    """将文本按指定长度切分成块"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

with open(output_pretrain_data, 'a', encoding='utf-8') as write_pre:
    with open(read_pretrain_data, 'r', encoding='utf-8') as read_pre:
        for line in tqdm(read_pre, desc=f"Processing", leave=False, unit="lines"):  # 添加行级别的进度条
            line = json.loads(line)
            text = line['text']
            chunks = split_text(text)
            for chunk in chunks:
                write_pre.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')

# 2 处理SFT数据
def convert_message(data):
    """
    将原始数据转换为标准格式
    """
    message = [
        {"role": "system", "content": "你是一个AI助手"},
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role': 'assistant', 'content': item['value']})
    return message

with open(output_sft_data, 'a', encoding='utf-8') as write_sft:
    with open(read_sft_data, 'r', encoding='utf-8') as read_sft:
        for item in tqdm(read_sft, desc="Processing", leave=False, unit="lines"):
            item = json.loads(item)
            message = convert_message(item['conversations'])
            write_sft.write(json.dumps(message, ensure_ascii=False) + '\n')