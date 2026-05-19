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


class PretrainProcessor:
    """对预训练原始语料做 token 级 packing：批量 tokenize、按 chunk_size 切片落盘，支持断点恢复。

    断点策略：
      - 每处理 checkpoint_interval 个 docs 写一次断点；
      - 状态包含：已处理 docs 数、输出文件字节位置、已写样本数、未成块的 token 缓冲；
      - 恢复时把输出文件 truncate 到记录的字节位置，去掉可能未完整写入的尾巴；
      - 跑完后删除断点；断点存在 = 未完成，可恢复。
    """

    DEFAULT_CHUNK_SIZE = 1024            # 与模型 max_seq_len 对齐
    DEFAULT_BATCH_SIZE = 5000            # 批量 tokenize 的批大小
    DEFAULT_CHECKPOINT_INTERVAL = 100_000  # 必须是 batch_size 的倍数

    def __init__(self,
                 input_path: str=read_pretrain_data,
                 output_path: str=output_pretrain_data,
                 tokenizer_dir: str=TOKENIZER_DIR,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL):
        assert checkpoint_interval % batch_size == 0, \
            "checkpoint_interval 必须是 batch_size 的整数倍，否则保存断点时 batch_texts 可能非空，状态不一致"
        self.input_path = input_path
        self.output_path = output_path
        self.tokenizer_dir = tokenizer_dir
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.ckpt_path = output_path + '.ckpt'

    # ---------- 内部工具 ----------
    def _flush_chunks(self, buf, write_fp):
        """从 buffer 中切出所有完整 chunk_size 块写盘，返回写出样本数。"""
        n_full = len(buf) // self.chunk_size
        for i in range(n_full):
            chunk = buf[i * self.chunk_size : (i + 1) * self.chunk_size]
            write_fp.write(json.dumps({'input_ids': chunk}) + '\n')
        del buf[:n_full * self.chunk_size]   # 原地删除，避免重新分配大列表
        return n_full

    def _save_checkpoint(self, docs_processed, output_bytes, n_samples, buffer):
        """原子写入断点文件：先写 .tmp 再 rename，避免半写崩溃。"""
        state = {
            'docs_processed': docs_processed,
            'output_bytes': output_bytes,
            'n_samples': n_samples,
            'buffer': buffer,
        }
        tmp = self.ckpt_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f)
        os.replace(tmp, self.ckpt_path)

    def _load_checkpoint(self):
        with open(self.ckpt_path, 'r') as f:
            return json.load(f)

    # ---------- 入口 ----------
    def run(self):
        output_exists = os.path.exists(self.output_path)
        ckpt_exists = os.path.exists(self.ckpt_path)

        if output_exists and not ckpt_exists:
            print(f"Skip pretrain: {self.output_path} already exists")
            return

        if not os.path.exists(self.tokenizer_dir):
            raise FileNotFoundError(
                f"Tokenizer not found at {self.tokenizer_dir}. "
                "Please run train_tokenizer.py first."
            )
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        eos_id = tokenizer.eos_token_id

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        if ckpt_exists:
            state = self._load_checkpoint()
            docs_processed = state['docs_processed']
            n_samples = state['n_samples']
            buffer = state['buffer']
            output_bytes = state['output_bytes']
            # 截断输出文件，去掉断点之后可能未完整写入的内容
            with open(self.output_path, 'r+b') as f:
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

        with open(self.output_path, write_mode, encoding='utf-8') as write_pre, \
             open(self.input_path, 'r', encoding='utf-8') as read_pre:

            # 跳过已处理的行（快速顺序读，无 json 解析开销）
            for _ in range(docs_processed):
                read_pre.readline()

            pbar = tqdm(read_pre, desc="Packing pretrain", unit="docs",
                        initial=docs_processed)
            for line in pbar:
                batch_texts.append(json.loads(line)['text'])
                docs_processed += 1

                if len(batch_texts) >= self.batch_size:
                    # 批量 tokenize：HF fast tokenizer 内部 Rust 多线程，比逐条快 5-10 倍
                    encs = tokenizer(batch_texts, add_special_tokens=False)['input_ids']
                    for ids in encs:
                        buffer.extend(ids)
                        buffer.append(eos_id)
                    batch_texts.clear()

                    if len(buffer) >= self.chunk_size:
                        n_samples += self._flush_chunks(buffer, write_pre)

                    # 周期性保存断点（此刻 batch_texts 已清空，状态一致）
                    if docs_processed % self.checkpoint_interval == 0:
                        write_pre.flush()
                        os.fsync(write_pre.fileno())
                        self._save_checkpoint(
                            docs_processed,
                            os.path.getsize(self.output_path),
                            n_samples,
                            buffer,
                        )

            # 处理尾部不足 batch_size 的剩余文本
            if batch_texts:
                encs = tokenizer(batch_texts, add_special_tokens=False)['input_ids']
                for ids in encs:
                    buffer.extend(ids)
                    buffer.append(eos_id)
                n_samples += self._flush_chunks(buffer, write_pre)

        # 全部完成，删除断点
        if os.path.exists(self.ckpt_path):
            os.remove(self.ckpt_path)

        print(f"Packed {n_samples} samples ({self.chunk_size} tokens each). "
              f"Discarded {len(buffer)} tail tokens.")


class SFTProcessor:
    """把 SFT 原始数据（ShareGPT 风格 conversations）转成 HuggingFace 风格 messages 的 jsonl。

    输出每行：{"messages": [{"role": "system", ...}, {"role": "user", ...}, ...]}。
    无效样本（空对话、内容缺失、最后一条不是 assistant、user/assistant 不交替等）一律丢弃。
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是 MyLLM，一个由用户从零训练的中文人工智能助手。"
        "请用清晰、准确、有条理的中文回答用户问题；"
        "在不确定时如实说明，不要编造事实；"
        "涉及代码、数学或推理时分步给出过程，最后给出明确结论。"
    )
    DEFAULT_MAX_LEN = 1024   # 与模型 max_seq_len 对齐；超长样本直接丢弃

    def __init__(self,
                 input_path: str = read_sft_data,
                 output_path: str = output_sft_data,
                 tokenizer_dir: str = TOKENIZER_DIR,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 max_len: int = DEFAULT_MAX_LEN):
        self.input_path = input_path
        self.output_path = output_path
        self.tokenizer_dir = tokenizer_dir
        self.system_prompt = system_prompt
        self.max_len = max_len
        # 阶段 2 的输出路径：在 output_path 文件名后加 _tokenized 后缀
        self.tokenized_path = output_path.replace('.jsonl', '_tokenized.jsonl')

    # ---------- 内部工具 ----------
    def _convert_message(self, data):
        """单条 conversations -> messages，无效返回 None。"""
        if not data:
            return None

        message = [{"role": "system", "content": self.system_prompt}]
        for item in data:
            role = item.get('from')
            content = item.get('value')
            if not content:
                return None
            if role == 'human':
                message.append({'role': 'user', 'content': content})
            elif role in ('assistant', 'gpt'):
                message.append({'role': 'assistant', 'content': content})
            # 未知 role 单条忽略，最后由整体结构校验把关

        # 至少要有 system + user + assistant 三条，且最后一条必须是 assistant
        if len(message) < 3 or message[-1]['role'] != 'assistant':
            return None
        # user / assistant 必须严格交替
        expected = 'user'
        for m in message[1:]:
            if m['role'] != expected:
                return None
            expected = 'assistant' if expected == 'user' else 'user'

        return message
    
    def _tokenize_one(self, tokenizer, messages):
        """单条 messages -> (input_ids, labels)。超长或前缀对不齐返回 None。

        多轮支持：遍历每一个 assistant 消息，分别渲染 "到该消息之前" 与 "到该消息为止"
        两个字符串，差集 token 区间即这一轮 assistant 的 content + <|im_end|>，
        填回 labels；其余位置（system / user / 中间空隙）保持 -100。
        """
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        full_ids = tokenizer(full_text, add_special_tokens=False)['input_ids']

        if len(full_ids) > self.max_len:
            return None

        labels = [-100] * len(full_ids)

        for i, m in enumerate(messages):
            if m['role'] != 'assistant':
                continue
            # 到第 i 条 assistant 之前：用 generation_prompt 让结尾停在 "<|im_start|>assistant\n"
            before_text = tokenizer.apply_chat_template(
                messages[:i], tokenize=False, add_generation_prompt=True,
            )
            # 到第 i 条 assistant 为止：不加 generation_prompt，正常以 "<|im_end|>\n" 收尾
            upto_text = tokenizer.apply_chat_template(
                messages[:i + 1], tokenize=False, add_generation_prompt=False,
            )
            before_ids = tokenizer(before_text, add_special_tokens=False)['input_ids']
            upto_ids   = tokenizer(upto_text,   add_special_tokens=False)['input_ids']

            # 防御：必须满足  before_ids  ⊏  upto_ids  ⊏  full_ids  且严格变长
            if (len(before_ids) >= len(upto_ids)
                    or len(upto_ids) > len(full_ids)
                    or full_ids[:len(before_ids)] != before_ids
                    or full_ids[:len(upto_ids)]   != upto_ids):
                return None

            start, end = len(before_ids), len(upto_ids)
            labels[start:end] = full_ids[start:end]

        # 没有任何 assistant 区间被填上 -> 整条样本不会贡献 loss，直接丢弃
        if all(l == -100 for l in labels):
            return None

        return full_ids, labels

    def _tokenize_file(self):
        """阶段 2：读 messages jsonl，输出已 tokenize 的 (input_ids, labels) jsonl。"""
        if os.path.exists(self.tokenized_path):
            print(f"Skip tokenization: {self.tokenized_path} already exists")
            return
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(
                f"Messages file {self.output_path} 不存在，先运行 _convert_file()。"
            )

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)

        n_kept, n_too_long = 0, 0
        with open(self.output_path,    'r', encoding='utf-8') as rf, \
             open(self.tokenized_path, 'w', encoding='utf-8') as wf:
            for line in tqdm(rf, desc="Tokenizing sft", unit="msg"):
                messages = json.loads(line)['messages']
                result = self._tokenize_one(tokenizer, messages)
                if result is None:
                    # 简化处理：长度超限和前缀对不齐都归到 too_long 计数
                    # 若想区分可在 _tokenize_one 返回不同标记
                    n_too_long += 1
                    continue
                input_ids, labels = result
                wf.write(json.dumps({'input_ids': input_ids, 'labels': labels}) + '\n')
                n_kept += 1
        print(f"Tokenized SFT done: kept={n_kept}, dropped={n_too_long} (too long / prefix mismatch)")

    def _convert_file(self):
        """阶段 1：原始 conversations -> messages jsonl。"""
        if os.path.exists(self.output_path):
            print(f"Skip sft: {self.output_path} already exists")
            return

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        n_kept, n_skipped = 0, 0
        with open(self.output_path, 'w', encoding='utf-8') as write_sft, \
             open(self.input_path,  'r', encoding='utf-8') as read_sft:
            for line in tqdm(read_sft, desc="Processing sft", leave=True, unit="lines"):
                item = json.loads(line)
                message = self._convert_message(item.get('conversations'))
                if message is None:
                    n_skipped += 1
                    continue
                write_sft.write(json.dumps({"messages": message}, ensure_ascii=False) + '\n')
                n_kept += 1
        print(f"SFT done: kept={n_kept}, skipped={n_skipped}")

    # ---------- 入口 ----------
    def run(self):
        """两阶段流水线：原始格式 -> messages jsonl -> tokenized jsonl。"""
        self._convert_file()
        self._tokenize_file()


if __name__ == "__main__":
    # PretrainProcessor(
    #     input_path=read_pretrain_data,
    #     output_path=output_pretrain_data,
    #     tokenizer_dir=TOKENIZER_DIR,
    # ).run()

    SFTProcessor(
        input_path=read_sft_data,
        output_path=output_sft_data,
    ).run()
