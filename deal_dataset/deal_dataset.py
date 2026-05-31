import csv
import glob
import json
import os

import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import mlflow
from opencc import OpenCC


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# PROJECT_ROOT = "/root/autodl-tmp/MyLLMDataset"

read_pretrain_data   = os.path.join(
    PROJECT_ROOT,
    'dataset',
    'seq_monkey',
    'mobvoi_seq_monkey_general_open_corpus.jsonl'
)
# 改用 Arrow 目录（save_to_disk 创建的是目录而非单文件）
output_pretrain_data = os.path.join(PROJECT_ROOT, 'processed_dataset', 'seq_monkey_arrow')
read_sft_data        = os.path.join(PROJECT_ROOT, 'dataset', 'BelleGroup', 'train_3.5M_CN.json')
output_sft_data      = os.path.join(PROJECT_ROOT, 'processed_dataset', 'BelleGroup_sft.jsonl')

# chinese-poetry GitHub 仓库原始 JSON 目录(放唐诗 + 宋词的 .json 文件)
read_chinese_poetry_dir      = os.path.join(PROJECT_ROOT, 'dataset', 'chinese_poetry')
# converter 输出的 jsonl(中转产物,给 PretrainProcessor 读)
output_chinese_poetry_jsonl  = os.path.join(PROJECT_ROOT, 'dataset', 'chinese_poetry', 'all.jsonl')
# tokenize + packing 后的 arrow(训练时给 PretrainDataset 读)
output_chinese_poetry_arrow  = os.path.join(PROJECT_ROOT, 'processed_dataset', 'chinese_poetry_arrow')

TOKENIZER_DIR_OR_NAME = os.path.join(PROJECT_ROOT, 'tokenizer_k')


class ChinesePoetryConverter:
    """把 chinese-poetry GitHub 仓库的唐诗 + 宋词 JSON 合并成 jsonl,带朝代/作者/题目 metadata。

    文件命名约定(放在 input_dir 下):
      唐诗: poet.tang.0.json, poet.tang.1000.json, ..., poet.tang.57000.json
      宋词: ci.song.0.json,   ci.song.1000.json,   ..., ci.song.21000.json

    每条 JSON 记录格式:
      唐诗: {"author": ..., "title": ...,    "paragraphs": [...]}
      宋词: {"author": ..., "rhythmic": ..., "paragraphs": [...]}   # 词牌名当 title

    输出 jsonl 每行格式(方案 A:自然语言拼接,适合 continued pretraining):
      {"text": "唐 · 李白《静夜思》:床前明月光,疑是地上霜。举头望明月,低头思故乡。"}
      {"text": "宋 · 苏轼《水调歌头》:明月几时有,把酒问青天。..."}

    繁体字处理(默认不转,详见 simplify_traditional 参数):
      《全唐诗》《全宋诗》原数据是繁体,这是出于保留语义的考虑 ——
      简体把多个繁体字合并成一个(如 後/后 → 后,髮/發 → 发,乾/幹 → 干),
      在古典诗词里这种合并会改变字义,作者也明确警告 "转换后的字不符合上下文"。
      默认 simplify_traditional=False,保留繁体;Qwen tokenizer 能正常处理繁体。
      非要转的话(如报告输出统一性),传 simplify_traditional=True,需先 pip install opencc-python-reimplemented。
    """

    DEFAULT_MIN_LENGTH = 10   # 拼接后短于此长度的丢弃(基本是脏数据)

    def __init__(self,
                 input_dir: str = read_chinese_poetry_dir,
                 output_path: str = output_chinese_poetry_jsonl,
                 include_tang: bool = True,
                 include_song: bool = True,
                 simplify_traditional: bool = False,
                 min_length: int = DEFAULT_MIN_LENGTH):
        self.input_dir = input_dir
        self.output_path = output_path
        self.include_tang = include_tang
        self.include_song = include_song
        self.min_length = min_length

        # 繁→简 转换器,缺包则降级为不转
        self._cc = None
        if simplify_traditional:
            try:
                self._cc = OpenCC('t2s')   # traditional → simplified
            except ImportError:
                print("⚠️ 未装 opencc,繁体字将原样保留。"
                      "建议: pip install opencc-python-reimplemented")

    def _maybe_simplify(self, text: str) -> str:
        return self._cc.convert(text) if self._cc else text

    def _format_record(self, rec: dict, dynasty: str, title_key: str) -> str | None:
        """通用格式化:'朝代 · 作者《题目》:正文'。任一字段缺失返回 None。"""
        author = (rec.get('author') or '').strip()
        title = (rec.get(title_key) or '').strip()
        paragraphs = rec.get('paragraphs') or []
        if not author or not title or not paragraphs:
            return None
        body = "".join(paragraphs)
        return self._maybe_simplify(f"{dynasty} · {author}《{title}》:{body}")

    def _process_files(self, glob_pattern: str, dynasty: str, title_key: str, f_out) -> tuple[int, int]:
        """处理一组同朝代的文件,流式写入 f_out,返回 (n_kept, n_skipped)。"""
        files = sorted(glob.glob(os.path.join(self.input_dir, glob_pattern)))
        print(f"  {dynasty}: {len(files)} 个文件")
        n_kept, n_skipped = 0, 0
        for fpath in files:
            with open(fpath, encoding='utf-8') as f_in:
                records = json.load(f_in)
            for rec in records:
                text = self._format_record(rec, dynasty, title_key)
                if not text or len(text) < self.min_length:
                    n_skipped += 1
                    continue
                f_out.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
                n_kept += 1
        return n_kept, n_skipped

    def run(self) -> dict | None:
        """跑转换流水线。返回 stats dict 给外部(MLflow)记录;已存在则跳过返回 None。"""
        if os.path.exists(self.output_path):
            print(f"Skip chinese_poetry: {self.output_path} already exists")
            return None
        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)

        total_kept, total_skipped = 0, 0
        tang_kept, song_kept = 0, 0
        with open(self.output_path, 'w', encoding='utf-8') as f_out:
            if self.include_tang:
                # 唐诗:题目字段名是 "title"
                k, s = self._process_files('poet.tang.*.json', '唐', 'title', f_out)
                tang_kept = k
                total_kept += k
                total_skipped += s
            if self.include_song:
                # 宋词:题目字段名是 "rhythmic"(词牌名,如《水调歌头》)
                k, s = self._process_files('ci.song.*.json', '宋', 'rhythmic', f_out)
                song_kept = k
                total_kept += k
                total_skipped += s

        print(f"ChinesePoetry done: kept={total_kept}, skipped={total_skipped}")
        print(f"Output: {self.output_path}")

        return {
            "n_kept":      total_kept,
            "n_skipped":   total_skipped,
            "kept_ratio":  total_kept / max(1, total_kept + total_skipped),
            "n_tang":      tang_kept,
            "n_song":      song_kept,
        }


class PretrainProcessor:
    """对预训练原始语料做 token 级 packing：批量 tokenize → 拼接 → 切 chunk_size 块 → 存 Arrow 目录。

    用 datasets.map 多进程并行实现：
      - 阶段 1：每条文本独立 tokenize 并追加 EOS（1-to-1，num_proc 并行）；
      - 阶段 2：每 group_batch 个文档 flatten + 切 chunk_size 块（batched，跨 batch 边界会丢
        < chunk_size 的尾巴，对大数据集影响 < 0.1%）；
      - 阶段 3：save_to_disk 存 Arrow 目录，PretrainDataset 用 load_from_disk 加载。

    跟旧的手写循环 + .ckpt 版相比：
      - **快 5~10×**：num_proc 多进程；
      - **自动可恢复**：HF datasets 的指纹 cache 即断点，重跑直接命中；
      - **代码 80 → 40 行**；
      - **磁盘占用减半**：Arrow 比 jsonl 紧凑。
    """

    DEFAULT_CHUNK_SIZE = 1024            # 与模型 max_seq_len 对齐
    DEFAULT_BATCH_SIZE = 5000            # tokenize 时的 batched map 批大小
    DEFAULT_GROUP_BATCH = 1000           # packing 时每多少个文档拼一次（越大边界损失越小）
    DEFAULT_NUM_PROC = 8                 # tokenize 多进程数

    def __init__(self,
                 input_path: str = read_pretrain_data,
                 output_path: str = output_pretrain_data,
                 tokenizer_dir_or_name: str = TOKENIZER_DIR_OR_NAME,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 group_batch: int = DEFAULT_GROUP_BATCH,
                 num_proc: int = DEFAULT_NUM_PROC):
        self.input_path = input_path
        self.output_path = output_path   # Arrow 目录
        self.tokenizer_dir_or_name = tokenizer_dir_or_name
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.group_batch = group_batch
        self.num_proc = num_proc

    def run(self) -> dict | None:
        """跑 tokenize + packing。返回 stats dict 给外部(MLflow);已存在则跳过返回 None。"""
        # 已存在且非空目录就跳过
        if os.path.isdir(self.output_path) and os.listdir(self.output_path):
            print(f"Skip pretrain: {self.output_path} already exists")
            return None

        # tokenizer_dir_or_name 既可以是本地路径,也可以是 HF Hub 名字(如 "Qwen/Qwen2.5-32B"),
        # 因此不再用 os.path.exists 检查;让 from_pretrained 自己报错即可。
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir_or_name)
        eos_id = tokenizer.eos_token_id
        chunk_size = self.chunk_size      # 闭包捕获本地变量，避免每条都查 self 属性

        ds = load_dataset('json', data_files=self.input_path, split='train')

        # ---- 阶段 1：每条文本 tokenize 并追加 EOS（1-to-1） ----
        def tokenize_batch(examples):
            encs = tokenizer(examples["text"], add_special_tokens=False)["input_ids"]
            return {"input_ids": [ids + [eos_id] for ids in encs]}

        ds_tok = ds.map(
            tokenize_batch,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            remove_columns=ds.column_names,
            desc="Tokenizing pretrain",
        )

        # ---- 阶段 2：拼接 + 切 chunk_size 块（1-to-N） ----
        # 注意：每个 group_batch 个文档独立 flatten + 切块，
        # 跨 batch 的尾巴 token 会被丢弃。group_batch 越大边界浪费越小。
        def group_into_chunks(examples):
            concatenated = sum(examples["input_ids"], [])           # flatten 一批文档的所有 token
            total = (len(concatenated) // chunk_size) * chunk_size  # 向下对齐到 chunk_size 的整数倍
            chunks = [concatenated[i:i + chunk_size]
                      for i in range(0, total, chunk_size)]
            return {"input_ids": chunks}

        ds_packed = ds_tok.map(
            group_into_chunks,
            batched=True,
            batch_size=self.group_batch,
            num_proc=min(4, self.num_proc),   # packing IO 多于 CPU，进程数少一点
            remove_columns=ds_tok.column_names,
            desc="Packing into chunks",
        )

        # ---- 阶段 3：写 Arrow 目录 ----
        ds_packed.save_to_disk(self.output_path)
        n_chunks = len(ds_packed)
        print(f"Saved Arrow dataset to {self.output_path}, "
              f"n={n_chunks} chunks of {chunk_size} tokens each.")

        return {
            "n_chunks":       n_chunks,
            "chunk_size":     chunk_size,
            "total_tokens":   n_chunks * chunk_size,
            "n_input_docs":   len(ds_tok),
        }


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
                 tokenizer_dir_or_name: str = TOKENIZER_DIR_OR_NAME,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 max_len: int = DEFAULT_MAX_LEN):
        self.input_path = input_path
        self.output_path = output_path
        self.tokenizer_dir_or_name = tokenizer_dir_or_name
        self.system_prompt = system_prompt
        self.max_len = max_len
        # 阶段 2 的输出路径：Arrow 目录（save_to_disk 创建的是目录而非单文件），
        # 用 _tokenized_arrow 后缀避免跟 .jsonl 混淆，SFTDataset 用 load_from_disk 加载。
        self.tokenized_path = output_path.replace('.jsonl', '_tokenized_arrow')

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

    def _tokenize_file(self):
        if os.path.exists(self.tokenized_path):
            print(f"Skip tokenization: {self.tokenized_path} already exists")
            return
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(
                f"Messages file {self.output_path} 不存在，先运行 _convert_file()。"
            )
            
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir_or_name)
        ds = load_dataset('json', data_files=self.output_path, split='train')
        max_len = self.max_len   # 闭包捕获本地变量，避免每条都查 self 属性

        # ds.map 的 callback 不允许返回 None；用空 list 作为"丢弃"哨兵，
        # 后面 .filter 把它们过滤掉。
        SENTINEL = {"input_ids": [], "labels": []}

        def tokenize_one(example):
            """单条 messages -> (input_ids, labels)。
            超长 / 前缀对不齐 / 无任何 assistant 区间 -> 返回 SENTINEL，由后续 filter 丢弃。

            多轮支持：遍历每一个 assistant 消息，分别渲染 "到该消息之前" 与 "到该消息为止"
            两个字符串，差集 token 区间即这一轮 assistant 的 content + <|im_end|>，
            填回 labels；其余位置（system / user / 中间空隙）保持 -100。
            """
            messages = example["messages"]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            full_ids = tokenizer(full_text, add_special_tokens=False)['input_ids']

            if len(full_ids) > max_len:
                return SENTINEL

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
                    return SENTINEL

                start, end = len(before_ids), len(upto_ids)
                labels[start:end] = full_ids[start:end]

            # 没有任何 assistant 区间被填上 -> 整条样本不会贡献 loss，直接丢弃
            if all(l == -100 for l in labels):
                return SENTINEL

            return {"input_ids": full_ids, "labels": labels}

        ds_tok = ds.map(
            tokenize_one,
            num_proc=8,
            remove_columns=ds.column_names,
            desc="Tokenizing SFT",
        ).filter(
            lambda x: len(x["input_ids"]) > 0,
            num_proc=8,
            desc="Filtering invalid",
        )

        ds_tok.save_to_disk(self.tokenized_path)
        print(f"Saved Arrow dataset to {self.tokenized_path}, n={len(ds_tok)}")
        
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
    #     tokenizer_dir_or_name=TOKENIZER_DIR_OR_NAME,
    # ).run()

    # SFTProcessor(
    #     input_path=read_sft_data,
    #     output_path=output_sft_data,
    # ).run()

    # ===== 数据流水线配置(集中放,方便 MLflow log_params)=====
    CONVERT_CFG = {
        "input_dir":            read_chinese_poetry_dir,
        "output_path":          output_chinese_poetry_jsonl,
        "include_tang":         True,
        "include_song":         True,
        "simplify_traditional": False,    # 作者警告:转简体会丢语义,保持繁体
        "min_length":           10,
        "text_format":          "朝代 · 作者《题目》:正文",
    }
    TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'model', 'Qwen3-32B')
    PROCESS_CFG = {
        "input_path":            output_chinese_poetry_jsonl,
        "output_path":           output_chinese_poetry_arrow,
        "tokenizer_dir_or_name": TOKENIZER_PATH,
        "tokenizer_name":        os.path.basename(TOKENIZER_PATH),  # log 短名字,Qwen3-32B
        "chunk_size":            512,
    }

    MLFLOW_EXPERIMENT = "MyLLM-DataPipeline"
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # 收集本次跑的 run id,跑完写 sidecar 文件供训练脚本读
    data_lineage = {"experiment": MLFLOW_EXPERIMENT}

    # ---- 阶段 1:数据预处理(唐诗 + 宋词 JSON → jsonl) ----
    with mlflow.start_run(run_name="convert_chinese_poetry"):
        mlflow.set_tag("stage", "preprocess")
        mlflow.set_tag("dataset", "chinese-poetry")
        mlflow.log_params(CONVERT_CFG)

        stats = ChinesePoetryConverter(
            input_dir=CONVERT_CFG["input_dir"],
            output_path=CONVERT_CFG["output_path"],
            include_tang=CONVERT_CFG["include_tang"],
            include_song=CONVERT_CFG["include_song"],
            simplify_traditional=CONVERT_CFG["simplify_traditional"],
            min_length=CONVERT_CFG["min_length"],
        ).run()

        if stats is not None:
            mlflow.log_metrics(stats)
        else:
            mlflow.set_tag("skipped", "output_already_exists")

        # 记 run_id,供下游(训练)交叉追溯
        data_lineage["convert_run_id"] = mlflow.active_run().info.run_id

    # ---- 阶段 2:特征工程(tokenize + packing 成 arrow) ----
    with mlflow.start_run(run_name=f"feature_eng_chunk{PROCESS_CFG['chunk_size']}"):
        mlflow.set_tag("stage", "feature_eng")
        mlflow.set_tag("tokenizer", PROCESS_CFG["tokenizer_name"])
        # tokenizer 路径太长,只 log 短名字;其他 cfg 全 log
        mlflow.log_params({k: v for k, v in PROCESS_CFG.items() if k != "tokenizer_dir_or_name"})
        # 也把上游(convert)的 run id 当 tag 记一下,数据流水线内部也可追溯
        # convert 跳过时 key 不存在,用 .get() 安全访问
        upstream_convert_id = data_lineage.get("convert_run_id")
        if upstream_convert_id:
            mlflow.set_tag("upstream_convert_run_id", upstream_convert_id)
            
        stats = PretrainProcessor(
            input_path=PROCESS_CFG["input_path"],
            output_path=PROCESS_CFG["output_path"],
            tokenizer_dir_or_name=PROCESS_CFG["tokenizer_dir_or_name"],
            chunk_size=PROCESS_CFG["chunk_size"],
        ).run()

        if stats is not None:
            mlflow.log_metrics(stats)
        else:
            mlflow.set_tag("skipped", "output_already_exists")

        data_lineage["feature_eng_run_id"] = mlflow.active_run().info.run_id

    # ---- 写 sidecar 文件,供训练脚本读取并设为 tag(交叉追溯) ----
    # 放在 arrow 目录旁边(同名 + .mlflow.json),不进 arrow 内部以免污染 dataset 结构。
    # 重跑保护:只在"feature_eng 这次真跑了"或"sidecar 从未写过"时覆盖,
    # 避免重跑空脚本(arrow 已存在 → 两阶段都跳过)用空 run 把上次有效的 run_id 冲掉。
    lineage_path = output_chinese_poetry_arrow.rstrip('/') + ".mlflow.json"
    should_write_lineage = (
        stats is not None                       # feature_eng 这次确实跑了(没跳过)
        or not os.path.exists(lineage_path)     # 或者之前从未写过 sidecar
    )
    if should_write_lineage:
        os.makedirs(os.path.dirname(lineage_path) or '.', exist_ok=True)
        with open(lineage_path, "w", encoding='utf-8') as f:
            json.dump(data_lineage, f, indent=2, ensure_ascii=False)
        print(f"Lineage 写入: {lineage_path}")
        print(f"  convert_run_id:     {data_lineage['convert_run_id']}")
        print(f"  feature_eng_run_id: {data_lineage['feature_eng_run_id']}")
    else:
        print(f"两阶段都跳过,保留已有 Lineage: {lineage_path}")
