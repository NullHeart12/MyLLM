import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = "/root/autodl-tmp/MyLLMDataset"

read_pretrain_data   = os.path.join(PROJECT_ROOT, 'dataset', 'mobvoi_seq_monkey_general_open_corpus.jsonl')
output_pretrain_data = os.path.join(PROJECT_ROOT, 'processed_dataset', 'seq_monkey.jsonl')
read_sft_data        = os.path.join(PROJECT_ROOT, 'dataset', 'BelleGroup', 'train_3.5M_CN.json')
output_sft_data      = os.path.join(PROJECT_ROOT, 'processed_dataset', 'BelleGroup_sft.jsonl')

TOKENIZER_DIR = os.path.join(PROJECT_ROOT, 'tokenizer_k')
CHUNK_SIZE = 1024   # 与模型 max_seq_len 对齐


BATCH_SIZE = 5000              # 批量 tokenize 的批大小
CHECKPOINT_INTERVAL = 100_000  # 每处理多少 docs 保存一次断点(必须是 BATCH_SIZE 的倍数)


def _flush_chunks(buf, write_fp):
    """从 buffer 中切出所有完整 CHUNK_SIZE 块写盘,返回写出样本数"""
    n_full = len(buf) // CHUNK_SIZE
    for i in range(n_full):
        chunk = buf[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        write_fp.write(json.dumps({'input_ids': chunk}) + '\n')
    del buf[:n_full * CHUNK_SIZE]   # 原地删除,避免重新分配大列表
    return n_full


def _ckpt_path(output_path):
    return output_path + '.ckpt'


def _save_checkpoint(ckpt_path, docs_processed, output_bytes, n_samples, buffer):
    """原子写入断点文件:先写 .tmp 再 rename,避免半写崩溃"""
    state = {
        'docs_processed': docs_processed,
        'output_bytes': output_bytes,
        'n_samples': n_samples,
        'buffer': buffer,
    }
    tmp = ckpt_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f)
    os.replace(tmp, ckpt_path)


def _load_checkpoint(ckpt_path):
    with open(ckpt_path, 'r') as f:
        return json.load(f)


def process_pretrain():
    """处理预训练数据(token 级 packing,批量 tokenize,支持断点恢复)

    断点策略:
      - 每处理 CHECKPOINT_INTERVAL 个 docs 保存一次状态
      - 状态:已处理 docs 数、输出文件字节位置、已写样本数、未成块的 token 缓冲
      - 恢复时把输出文件 truncate 到记录的字节位置(去掉可能未完整写入的尾巴)
      - 跑完后删除断点文件;断点存在 = 未完成,可恢复
    """
    ckpt_path = _ckpt_path(output_pretrain_data)
    output_exists = os.path.exists(output_pretrain_data)
    ckpt_exists = os.path.exists(ckpt_path)

    if output_exists and not ckpt_exists:
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

    if ckpt_exists:
        state = _load_checkpoint(ckpt_path)
        docs_processed = state['docs_processed']
        n_samples = state['n_samples']
        buffer = state['buffer']
        output_bytes = state['output_bytes']
        # 截断输出文件,去掉断点之后可能未完整写入的内容
        with open(output_pretrain_data, 'r+b') as f:
            f.truncate(output_bytes)
        print(f"Resuming: docs={docs_processed}, samples={n_samples}, "
              f"buffer={len(buffer)} tokens, output={output_bytes} bytes")
        write_mode = 'a'
    else:
        docs_processed = 0
        n_samples = 0
        buffer = []
        write_mode = 'w'

    batch_texts = []

    with open(output_pretrain_data, write_mode, encoding='utf-8') as write_pre, \
         open(read_pretrain_data, 'r', encoding='utf-8') as read_pre:

        # 跳过已处理的行(快速顺序读,无 json 解析开销)
        for _ in range(docs_processed):
            read_pre.readline()

        pbar = tqdm(read_pre, desc="Packing pretrain", unit="docs",
                    initial=docs_processed)
        for line in pbar:
            batch_texts.append(json.loads(line)['text'])
            docs_processed += 1

            if len(batch_texts) >= BATCH_SIZE:
                # 批量 tokenize:HF fast tokenizer 内部 Rust 多线程,比逐条快 5-10 倍
                encs = tokenizer(batch_texts, add_special_tokens=False)['input_ids']
                for ids in encs:
                    buffer.extend(ids)
                    buffer.append(eos_id)
                batch_texts.clear()

                if len(buffer) >= CHUNK_SIZE:
                    n_samples += _flush_chunks(buffer, write_pre)

                # 周期性保存断点(此刻 batch_texts 已清空,状态一致)
                if docs_processed % CHECKPOINT_INTERVAL == 0:
                    write_pre.flush()
                    os.fsync(write_pre.fileno())
                    _save_checkpoint(
                        ckpt_path,
                        docs_processed,
                        os.path.getsize(output_pretrain_data),
                        n_samples,
                        buffer,
                    )

        # 处理尾部不足 BATCH_SIZE 的剩余文本
        if batch_texts:
            encs = tokenizer(batch_texts, add_special_tokens=False)['input_ids']
            for ids in encs:
                buffer.extend(ids)
                buffer.append(eos_id)
            n_samples += _flush_chunks(buffer, write_pre)

    # 全部完成,删除断点
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

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
    # process_sft()
