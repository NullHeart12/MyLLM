import csv
import glob
import json
import os

# 必须在 import transformers / datasets 之前设
# fast tokenizer 内部用 Rust 线程并行,跟 Python multiprocessing 的 fork 模型冲突,
# 会导致 datasets.map(num_proc>1) 时 worker 莫名死亡。
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

# secret 数据集(从 HuggingFace 下载,目录下若干 *.json 文件)
read_secret_dir              = os.path.join(PROJECT_ROOT, 'dataset', 'secret')
output_secret_jsonl          = os.path.join(PROJECT_ROOT, 'dataset', 'secret', 'all.jsonl')
output_secret_arrow          = os.path.join(PROJECT_ROOT, 'processed_dataset', 'secret_arrow')

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

    DEFAULT_MIN_LENGTH = 10      # 拼接后短于此长度的丢弃(基本是脏数据)
    DEFAULT_MAX_BIO_LENGTH = 1100 # 作者生平超过此长度截断(防止 Song description 几千字撑爆训练分布)

    def __init__(self,
                 input_dir: str = read_chinese_poetry_dir,
                 output_path: str = output_chinese_poetry_jsonl,
                 include_tang: bool = True,
                 include_song: bool = True,
                 include_authors: bool = True,
                 simplify_traditional: bool = False,
                 min_length: int = DEFAULT_MIN_LENGTH,
                 max_bio_length: int = DEFAULT_MAX_BIO_LENGTH):
        self.input_dir = input_dir
        self.output_path = output_path
        self.include_tang = include_tang
        self.include_song = include_song
        self.include_authors = include_authors
        self.min_length = min_length
        self.max_bio_length = max_bio_length

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

    # 古籍缺字符号:整理者用 □ 标记原稿残缺/模糊不可辨识的字,
    # 也偶有 ▢、■、? 等变体。任一出现都说明这首诗的文本不完整,丢弃。
    _MISSING_CHAR_MARKERS = ('□', '▢', '■', '?', '?')

    def _format_record(self, rec: dict, dynasty: str, title_key: str) -> str | None:
        """通用格式化:'朝代 · 作者《题目》:正文'。任一字段缺失或正文含古籍缺字符号返回 None。
        有 tags 字段(如『唐诗三百首』『五言律诗』『田园』)的诗会额外拼上标签段。
        """
        author = (rec.get('author') or '').strip()
        title = (rec.get(title_key) or '').strip()
        paragraphs = rec.get('paragraphs') or []
        if not author or not title or not paragraphs:
            return None
        body = "".join(paragraphs)
        # 过滤古籍缺字样本(原稿残缺/不可辨识,训练价值低且会污染生成)
        if any(marker in body for marker in self._MISSING_CHAR_MARKERS):
            return None

        # 标签段:有 tags 才拼,顿号分隔;没有就整段省略,避免出现"标签：无"
        tags = rec.get('tags') or []
        tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
        tags_seg = f"标签：{'、'.join(tags)}；" if tags else ""

        if dynasty == "唐":
            return self._maybe_simplify(
                f"唐诗题目：《{title}》；作者：{author}；朝代：{dynasty}；{tags_seg}正文：{body}"
            )
        return self._maybe_simplify(
            f"宋词词牌名：《{title}》；作者：{author}；朝代：{dynasty}；{tags_seg}正文：{body}"
        )

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

    # 没有生平字段时的兜底文本(让模型至少学到「作者→朝代」的对应关系)
    _BIO_PLACEHOLDER = "暂无生平介绍"

    def _format_author_record(self, rec: dict, dynasty: str) -> str | None:
        """把 author bio JSON 一条记录格式化成自然语言文本。
        字段差异:
          唐: {'name': '...', 'desc': '...', 'id': '...'}
          宋: {'name': '...', 'description': '<长版,可能上千字>',
                'short_description': '<精简版,几百字>'}
        优先用 description(宋),fallback 到 short_description,然后截断 max_bio_length。
        没有任何生平字段时,写「暂无生平介绍」占位,仍保留作者→朝代的对应关系。
        """
        name = (rec.get('name') or '').strip()
        if not name:
            return None   # 连名字都没有就真的没用了

        if dynasty == "唐":
            bio = (rec.get('desc') or '').strip()
            role = "诗人"
        else:
            # 宋:description > short_description
            bio = (rec.get('description') or rec.get('short_description') or '').strip()
            role = "词人"

        # 缺字符号检查 — bio 含 □ 视同没有 bio,降级到占位
        if bio=="--" and any(marker in bio for marker in self._MISSING_CHAR_MARKERS):
            bio = ""

        if not bio:
            bio = self._BIO_PLACEHOLDER

        if len(bio) > self.max_bio_length:
            bio = bio[:self.max_bio_length].rstrip() + "……"

        return self._maybe_simplify(
            f"{dynasty}代{role}介绍。姓名：{name}；朝代：{dynasty}；生平：{bio}"
        )

    def _process_author_files(self, glob_pattern: str, dynasty: str, f_out) -> tuple[int, int]:
        """处理一组 author bio JSON 文件,流式写入 f_out。返回 (n_kept, n_skipped)。"""
        files = sorted(glob.glob(os.path.join(self.input_dir, glob_pattern)))
        print(f"  {dynasty} authors: {len(files)} 个文件")
        n_kept, n_skipped = 0, 0
        for fpath in files:
            with open(fpath, encoding='utf-8') as f_in:
                records = json.load(f_in)
            for rec in records:
                text = self._format_author_record(rec, dynasty)
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
        tang_authors, song_authors = 0, 0
        with open(self.output_path, 'w', encoding='utf-8') as f_out:
            # ===== 1. 作者简介(放在 jsonl 最前面)=====
            # 让模型在训练早期就先看到「朝代-作者-生平」的对应关系,
            # 之后看到具体诗词时能更好地把作者信号挂上去。
            # 兼容 author 与 authors 命名(用户可能两种都用过)
            if self.include_authors:
                k, s = self._process_author_files('author*.tang*.json', '唐', f_out)
                tang_authors = k
                total_kept += k
                total_skipped += s
                k, s = self._process_author_files('author*.song*.json', '宋', f_out)
                song_authors = k
                total_kept += k
                total_skipped += s

            # ===== 2. 唐诗 =====
            if self.include_tang:
                # 唐诗:题目字段名是 "title"
                k, s = self._process_files('poet.tang.*.json', '唐', 'title', f_out)
                tang_kept = k
                total_kept += k
                total_skipped += s

            # ===== 3. 宋词 =====
            if self.include_song:
                # 宋词:题目字段名是 "rhythmic"(词牌名,如《水调歌头》)
                k, s = self._process_files('ci.song.*.json', '宋', 'rhythmic', f_out)
                song_kept = k
                total_kept += k
                total_skipped += s

        print(f"ChinesePoetry done: kept={total_kept}, skipped={total_skipped}")
        print(f"Output: {self.output_path}")

        return {
            "n_kept":         total_kept,
            "n_skipped":      total_skipped,
            "kept_ratio":     total_kept / max(1, total_kept + total_skipped),
            "n_tang":         tang_kept,
            "n_song":         song_kept,
            "n_tang_authors": tang_authors,
            "n_song_authors": song_authors,
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


class SecretCollectionConverter:
    """把 `dataset/secret/` 下的所有 *.json 文件合并成一个 all.jsonl,
    供 PretrainProcessor 做后续 tokenize + packing。

    JSON 结构兼容三种情况(数据集源头不一,做防御性解析):
      1. List[str]            — 每个字符串就是一条 text
      2. List[Dict]           — 在 dict 里依次找 text / content / story / body 等字段
      3. Dict                 — 同 2,但只取这一条
    其他结构(如嵌套)直接跳过并计数到 n_skipped。

    输出 jsonl 每行 {"text": "..."}, 跟 ChinesePoetryConverter 对齐,
    PretrainProcessor 完全不用改就能消费。
    """

    DEFAULT_MIN_LENGTH = 50      # 短于此长度的样本基本是元数据/标题,丢弃
    DEFAULT_MAX_LENGTH = 100_000 # 单条文本超过此长度截断,防止极端样本撑爆 packing
    # 按优先级尝试这些字段名找正文(常见数据集字段习惯)
    TEXT_FIELDS = ('text', 'content', 'story', 'body', 'article', 'novel', 'output')

    def __init__(self,
                 input_dir: str = read_secret_dir,
                 output_path: str = output_secret_jsonl,
                 min_length: int = DEFAULT_MIN_LENGTH,
                 max_length: int = DEFAULT_MAX_LENGTH,
                 glob_pattern: str = "*.json"):
        self.input_dir = input_dir
        self.output_path = output_path
        self.min_length = min_length
        self.max_length = max_length
        self.glob_pattern = glob_pattern

    def _extract_text(self, rec) -> str | None:
        """从一条记录(可能是 str 或 dict)里抽取正文文本。"""
        if isinstance(rec, str):
            return rec
        if isinstance(rec, dict):
            for field in self.TEXT_FIELDS:
                val = rec.get(field)
                if isinstance(val, str) and val.strip():
                    return val
        return None

    def _process_one_file(self, fpath: str, f_out) -> tuple[int, int]:
        """处理单个 JSON 文件,返回 (n_kept, n_skipped)。"""
        try:
            with open(fpath, encoding='utf-8') as f_in:
                data = json.load(f_in)
        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON 解析失败,跳过:{fpath} ({e})")
            return 0, 0

        # 三种顶层结构统一成 List 遍历
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            print(f"  ⚠️ 未知顶层结构 {type(data).__name__},跳过:{fpath}")
            return 0, 0

        n_kept, n_skipped = 0, 0
        for rec in records:
            text = self._extract_text(rec)
            if text is None:
                n_skipped += 1
                continue
            text = text.strip()
            if len(text) < self.min_length:
                n_skipped += 1
                continue
            # 超长截断 + 标记(让模型学到尾部省略也是 OK 的)
            if len(text) > self.max_length:
                text = text[:self.max_length].rstrip() + "……"
            f_out.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
            n_kept += 1
        return n_kept, n_skipped

    def run(self) -> dict | None:
        """合并所有 *.json 到 all.jsonl。返回 stats dict 给 MLflow;已存在则跳过返回 None。"""
        if os.path.exists(self.output_path):
            print(f"Skip secret: {self.output_path} already exists")
            return None
        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)

        files = sorted(glob.glob(os.path.join(self.input_dir, self.glob_pattern)))
        # 跳过 all.jsonl 本身,以防 glob 把上次产物吃进来
        files = [f for f in files if os.path.basename(f) != os.path.basename(self.output_path)]
        print(f"Secret: 扫到 {len(files)} 个 JSON 文件")

        total_kept, total_skipped = 0, 0
        with open(self.output_path, 'w', encoding='utf-8') as f_out:
            for fpath in files:
                k, s = self._process_one_file(fpath, f_out)
                total_kept += k
                total_skipped += s

        print(f"Secret done: kept={total_kept}, skipped={total_skipped}")
        print(f"Output: {self.output_path}")

        return {
            "n_kept":      total_kept,
            "n_skipped":   total_skipped,
            "kept_ratio":  total_kept / max(1, total_kept + total_skipped),
            "n_files":     len(files),
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
    # CONVERT_CFG = {
    #     "input_dir":            read_chinese_poetry_dir,
    #     "output_path":          output_chinese_poetry_jsonl,
    #     "include_tang":         True,
    #     "include_song":         True,
    #     "include_authors":      True,     # NEW: 把 author bio 一起作为训练样本
    #     "simplify_traditional": True,    # 作者警告:转简体会丢语义,保持繁体
    #     "min_length":           10,
    #     "max_bio_length":       1100,      # 作者生平超过此长度截断,防止训练分布被长 bio 主导
    #     "text_format":          "朝代 · 作者《题目》:正文 + 作者简介",
    # }
    # TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'model', 'Qwen3-8B')
    # PROCESS_CFG = {
    #     "input_path":            output_chinese_poetry_jsonl,
    #     "output_path":           output_chinese_poetry_arrow,
    #     "tokenizer_dir_or_name": TOKENIZER_PATH,
    #     "tokenizer_name":        os.path.basename(TOKENIZER_PATH),  # log 短名字,Qwen3-32B
    #     "chunk_size":            512,
    # }

    # MLFLOW_EXPERIMENT = "MyLLM-DataPipeline"
    # mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # # 收集本次跑的 run id,跑完写 sidecar 文件供训练脚本读
    # data_lineage = {"experiment": MLFLOW_EXPERIMENT}

    # # ---- 阶段 1:数据预处理(唐诗 + 宋词 JSON → jsonl) ----
    # with mlflow.start_run(run_name="convert_chinese_poetry"):
    #     mlflow.set_tag("stage", "preprocess")
    #     mlflow.set_tag("dataset", "chinese-poetry")
    #     mlflow.log_params(CONVERT_CFG)

    #     stats = ChinesePoetryConverter(
    #         input_dir=CONVERT_CFG["input_dir"],
    #         output_path=CONVERT_CFG["output_path"],
    #         include_tang=CONVERT_CFG["include_tang"],
    #         include_song=CONVERT_CFG["include_song"],
    #         include_authors=CONVERT_CFG["include_authors"],
    #         simplify_traditional=CONVERT_CFG["simplify_traditional"],
    #         min_length=CONVERT_CFG["min_length"],
    #         max_bio_length=CONVERT_CFG["max_bio_length"],
    #     ).run()

    #     if stats is not None:
    #         mlflow.log_metrics(stats)
    #     else:
    #         mlflow.set_tag("skipped", "output_already_exists")

    #     # 记 run_id,供下游(训练)交叉追溯
    #     data_lineage["convert_run_id"] = mlflow.active_run().info.run_id

    # # ---- 阶段 2:特征工程(tokenize + packing 成 arrow) ----
    # with mlflow.start_run(run_name=f"feature_eng_chunk{PROCESS_CFG['chunk_size']}"):
    #     mlflow.set_tag("stage", "feature_eng")
    #     mlflow.set_tag("tokenizer", PROCESS_CFG["tokenizer_name"])
    #     # tokenizer 路径太长,只 log 短名字;其他 cfg 全 log
    #     mlflow.log_params({k: v for k, v in PROCESS_CFG.items() if k != "tokenizer_dir_or_name"})
    #     # 也把上游(convert)的 run id 当 tag 记一下,数据流水线内部也可追溯
    #     # convert 跳过时 key 不存在,用 .get() 安全访问
    #     upstream_convert_id = data_lineage.get("convert_run_id")
    #     if upstream_convert_id:
    #         mlflow.set_tag("upstream_convert_run_id", upstream_convert_id)
            
    #     stats = PretrainProcessor(
    #         input_path=PROCESS_CFG["input_path"],
    #         output_path=PROCESS_CFG["output_path"],
    #         tokenizer_dir_or_name=PROCESS_CFG["tokenizer_dir_or_name"],
    #         chunk_size=PROCESS_CFG["chunk_size"],
    #     ).run()

    #     if stats is not None:
    #         mlflow.log_metrics(stats)
    #     else:
    #         mlflow.set_tag("skipped", "output_already_exists")

    #     data_lineage["feature_eng_run_id"] = mlflow.active_run().info.run_id

    # # ---- 写 sidecar 文件,供训练脚本读取并设为 tag(交叉追溯) ----
    # # 放在 arrow 目录旁边(同名 + .mlflow.json),不进 arrow 内部以免污染 dataset 结构。
    # # 重跑保护:只在"feature_eng 这次真跑了"或"sidecar 从未写过"时覆盖,
    # # 避免重跑空脚本(arrow 已存在 → 两阶段都跳过)用空 run 把上次有效的 run_id 冲掉。
    # lineage_path = output_chinese_poetry_arrow.rstrip('/') + ".mlflow.json"
    # should_write_lineage = (
    #     stats is not None                       # feature_eng 这次确实跑了(没跳过)
    #     or not os.path.exists(lineage_path)     # 或者之前从未写过 sidecar
    # )
    # if should_write_lineage:
    #     os.makedirs(os.path.dirname(lineage_path) or '.', exist_ok=True)
    #     with open(lineage_path, "w", encoding='utf-8') as f:
    #         json.dump(data_lineage, f, indent=2, ensure_ascii=False)
    #     print(f"Lineage 写入: {lineage_path}")
    #     print(f"  convert_run_id:     {data_lineage['convert_run_id']}")
    #     print(f"  feature_eng_run_id: {data_lineage['feature_eng_run_id']}")
    # else:
    #     print(f"两阶段都跳过,保留已有 Lineage: {lineage_path}")


    # ====================================================================
    # ===== Secret 数据流水线(/dataset/secret/*.json → arrow)         =====
    # ====================================================================
    SECRET_CONVERT_CFG = {
        "input_dir":   read_secret_dir,
        "output_path": output_secret_jsonl,
        "min_length":  50,        # 短于 50 字的样本(标题/元数据)丢弃
        "max_length":  200_000,   # 单条文本超过 10w 字截断
        "glob_pattern": "*.json",
    }
    TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'model', 'Qwen3-8B')
    SECRET_PROCESS_CFG = {
        "input_path":            output_secret_jsonl,
        "output_path":           output_secret_arrow,
        "tokenizer_dir_or_name": TOKENIZER_PATH,
        "tokenizer_name":        os.path.basename(TOKENIZER_PATH),
        "chunk_size":            4096,
        # secret 数据集单条样本长(动辄几万字),tokenize 时 worker 容易 OOM。
        # 把 num_proc 砍到 2、batch_size 砍到 200,牺牲速度换稳定。
        # 跑成功后想加速,再慢慢往上调。
        "num_proc":              2,
        "batch_size":            200,
    }

    SECRET_MLFLOW_EXPERIMENT = "MyLLM-DataPipeline"   # 共享同一个 experiment,run name 区分
    mlflow.set_experiment(SECRET_MLFLOW_EXPERIMENT)

    secret_lineage = {"experiment": SECRET_MLFLOW_EXPERIMENT, "dataset": "secret"}

    # ---- 阶段 1:convert(多 JSON 文件 → 单 jsonl) ----
    with mlflow.start_run(run_name="convert_secret"):
        mlflow.set_tag("stage", "preprocess")
        mlflow.set_tag("dataset", "secret")
        mlflow.log_params(SECRET_CONVERT_CFG)

        secret_stats = SecretCollectionConverter(
            input_dir=SECRET_CONVERT_CFG["input_dir"],
            output_path=SECRET_CONVERT_CFG["output_path"],
            min_length=SECRET_CONVERT_CFG["min_length"],
            max_length=SECRET_CONVERT_CFG["max_length"],
            glob_pattern=SECRET_CONVERT_CFG["glob_pattern"],
        ).run()

        if secret_stats is not None:
            mlflow.log_metrics(secret_stats)
        else:
            mlflow.set_tag("skipped", "output_already_exists")

        secret_lineage["convert_run_id"] = mlflow.active_run().info.run_id

    # ---- 阶段 2:feature_eng(tokenize + packing 成 arrow) ----
    with mlflow.start_run(run_name=f"feature_eng_secret_chunk{SECRET_PROCESS_CFG['chunk_size']}"):
        mlflow.set_tag("stage", "feature_eng")
        mlflow.set_tag("dataset", "secret")
        mlflow.set_tag("tokenizer", SECRET_PROCESS_CFG["tokenizer_name"])
        mlflow.log_params({k: v for k, v in SECRET_PROCESS_CFG.items()
                           if k != "tokenizer_dir_or_name"})

        upstream_id = secret_lineage.get("convert_run_id")
        if upstream_id:
            mlflow.set_tag("upstream_convert_run_id", upstream_id)

        secret_stats = PretrainProcessor(
            input_path=SECRET_PROCESS_CFG["input_path"],
            output_path=SECRET_PROCESS_CFG["output_path"],
            tokenizer_dir_or_name=SECRET_PROCESS_CFG["tokenizer_dir_or_name"],
            chunk_size=SECRET_PROCESS_CFG["chunk_size"],
            num_proc=SECRET_PROCESS_CFG["num_proc"],
            batch_size=SECRET_PROCESS_CFG["batch_size"],
        ).run()

        if secret_stats is not None:
            mlflow.log_metrics(secret_stats)
        else:
            mlflow.set_tag("skipped", "output_already_exists")

        secret_lineage["feature_eng_run_id"] = mlflow.active_run().info.run_id

    # ---- 写 sidecar(给下游训练 / 评估读) ----
    # 重跑保护跟 chinese-poetry 那段一致:feature_eng 真跑了 或 之前从未写过 才覆盖
    secret_lineage_path = output_secret_arrow.rstrip('/') + ".mlflow.json"
    should_write_secret_lineage = (
        secret_stats is not None
        or not os.path.exists(secret_lineage_path)
    )
    if should_write_secret_lineage:
        os.makedirs(os.path.dirname(secret_lineage_path) or '.', exist_ok=True)
        with open(secret_lineage_path, "w", encoding='utf-8') as f:
            json.dump(secret_lineage, f, indent=2, ensure_ascii=False)
        print(f"Secret lineage 写入: {secret_lineage_path}")
        print(f"  convert_run_id:     {secret_lineage.get('convert_run_id')}")
        print(f"  feature_eng_run_id: {secret_lineage.get('feature_eng_run_id')}")
    else:
        print(f"两阶段都跳过,保留已有 Lineage: {secret_lineage_path}")
