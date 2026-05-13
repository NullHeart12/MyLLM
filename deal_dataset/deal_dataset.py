import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

read_pretrain_data   = os.path.join(PROJECT_ROOT, 'dataset', 'mobvoi_seq_monkey_general_open_corpus.jsonl')
output_pretrain_data = os.path.join(PROJECT_ROOT, 'processed_dataset', 'seq_monkey.jsonl')
read_sft_data        = os.path.join(PROJECT_ROOT, 'dataset', 'BelleGroup', 'train_3.5M_CN.json')
output_sft_data      = os.path.join(PROJECT_ROOT, 'processed_dataset', 'BelleGroup_sft.jsonl')

TOKENIZER_DIR = os.path.join(PROJECT_ROOT, 'tokenizer_k')
CHUNK_SIZE = 512   # 与模型 max_seq_len 对齐


def process_pretrain():
    """处理预训练数据(token 级 packing)"""
    if os.path.exists(output_pretrain_data):
        print(f"Skip pretrain: {output_pretrain_data} already exists")
        return

    if not os.path.exists(TOKENIZER_DIR):
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_DIR}. "
            "Please run train_tokenizer.py first."
        )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    eos_id = tokenizer.eos_token_id

    os.makedirs(os.path.dirname(output_pretrain_data), exist_ok=True)

    buffer = []        # 跨文档的 token 缓冲区
    n_samples = 0      # 已写出的样本数

    with open(output_pretrain_data, 'w', encoding='utf-8') as write_pre:
        with open(read_pretrain_data, 'r', encoding='utf-8') as read_pre:
            for line in tqdm(read_pre, desc="Packing pretrain", unit="docs"):
                line = json.loads(line)
                ids = tokenizer(line['text'], add_special_tokens=False)['input_ids']
                buffer.extend(ids)
                buffer.append(eos_id)   # 文档之间用 EOS 分隔

                # 缓冲区累积到 ≥1 个完整块时,一次性切出所有完整块
                if len(buffer) >= CHUNK_SIZE:
                    n_full = len(buffer) // CHUNK_SIZE
                    for i in range(n_full):
                        chunk = buffer[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
                        write_pre.write(json.dumps({'input_ids': chunk}) + '\n')
                        n_samples += 1
                    buffer = buffer[n_full * CHUNK_SIZE:]   # 只剩不足一个块的尾巴

    print(f"Packed {n_samples} samples ({CHUNK_SIZE} tokens each). "
          f"Discarded {len(buffer)} tail tokens.")


def convert_message(data):
    """将原始 SFT 数据转换为标准 messages 格式"""
    message = [
        {"role": "system", "content": "你是一个AI助手"},
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role': 'assistant', 'content': item['value']})
    return message


def process_sft():
    """处理 SFT 数据"""
    if os.path.exists(output_sft_data):
        print(f"Skip sft: {output_sft_data} already exists")
        return

    os.makedirs(os.path.dirname(output_sft_data), exist_ok=True)

    with open(output_sft_data, 'w', encoding='utf-8') as write_sft:
        with open(read_sft_data, 'r', encoding='utf-8') as read_sft:
            for item in tqdm(read_sft, desc="Processing sft", leave=False, unit="lines"):
                item = json.loads(item)
                message = convert_message(item['conversations'])
                write_sft.write(json.dumps(message, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    process_pretrain()
    process_sft()
