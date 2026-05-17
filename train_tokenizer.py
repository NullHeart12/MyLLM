import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator, Optional

SPECIAL_TOKENS_NUM = 64

# 完整 Qwen 风格 chat template,支持 tools / think / tool_response 等
# 用相邻字符串字面量拼接,每行一段 Jinja 逻辑行,结尾的 \n 是 Jinja 输出里需要的换行
CHAT_TEMPLATE = (
    "{%- if tools %}\n"
    "    {{- '<|im_start|>system\\n' }}\n"
    "    {%- if messages[0].role == 'system' %}\n"
    "        {{- messages[0].content + '\\n\\n' }}\n"
    "    {%- endif %}\n"
    "    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n"
    "    {%- for tool in tools %}\n"
    "        {{- \"\\n\" }}\n"
    "        {{- tool | tojson }}\n"
    "    {%- endfor %}\n"
    "    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n"
    "{%- else %}\n"
    "    {%- if messages[0].role == 'system' %}\n"
    "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}\n"
    "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n"
    "{%- for message in messages[::-1] %}\n"
    "    {%- set index = (messages|length - 1) - loop.index0 %}\n"
    "    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n"
    "        {%- set ns.multi_step_tool = false %}\n"
    "        {%- set ns.last_query_index = index %}\n"
    "    {%- endif %}\n"
    "{%- endfor %}\n"
    "{%- for message in messages %}\n"
    "    {%- if message.content is string %}\n"
    "        {%- set content = message.content %}\n"
    "    {%- else %}\n"
    "        {%- set content = '' %}\n"
    "    {%- endif %}\n"
    "    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n"
    "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n"
    "    {%- elif message.role == \"assistant\" %}\n"
    "        {%- set reasoning_content = '' %}\n"
    "        {%- if message.reasoning_content is string %}\n"
    "            {%- set reasoning_content = message.reasoning_content %}\n"
    "        {%- else %}\n"
    "            {%- if '</think>' in content %}\n"
    "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n"
    "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n"
    "            {%- endif %}\n"
    "        {%- endif %}\n"
    "        {%- if true %}\n"
    "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n"
    "        {%- endif %}\n"
    "        {%- if message.tool_calls %}\n"
    "            {%- for tool_call in message.tool_calls %}\n"
    "                {%- if (loop.first and content) or (not loop.first) %}\n"
    "                    {{- '\\n' }}\n"
    "                {%- endif %}\n"
    "                {%- if tool_call.function %}\n"
    "                    {%- set tool_call = tool_call.function %}\n"
    "                {%- endif %}\n"
    "                {{- '<tool_call>\\n{\"name\": \"' }}\n"
    "                {{- tool_call.name }}\n"
    "                {{- '\", \"arguments\": ' }}\n"
    "                {%- if tool_call.arguments is string %}\n"
    "                    {{- tool_call.arguments }}\n"
    "                {%- else %}\n"
    "                    {{- tool_call.arguments | tojson }}\n"
    "                {%- endif %}\n"
    "                {{- '}\\n</tool_call>' }}\n"
    "            {%- endfor %}\n"
    "        {%- endif %}\n"
    "        {{- '<|im_end|>\\n' }}\n"
    "    {%- elif message.role == \"tool\" %}\n"
    "        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n"
    "            {{- '<|im_start|>user' }}\n"
    "        {%- endif %}\n"
    "        {{- '\\n<tool_response>\\n' }}\n"
    "        {{- content }}\n"
    "        {{- '\\n</tool_response>' }}\n"
    "        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n"
    "            {{- '<|im_end|>\\n' }}\n"
    "        {%- endif %}\n"
    "    {%- endif %}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "    {%- if open_thinking is defined and open_thinking is true %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}"
)

def read_texts_from_jsonl(
    file_path: str, 
    max_lines: Optional[int] = None
) -> Generator[str, None, None]:
    """读取JSONL文件并安全提取文本数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if max_lines is not None and line_num >= max_lines:
                break
            try:
                data = json.loads(line)
                if 'text' not in data:
                    raise KeyError(f"Missing 'text' field in line {line_num}")
                yield data['text']
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line {line_num}")
                continue
            except KeyError as e:
                print(e)
                continue

def train_tokenizer(
    data_path: str,
    save_dir: str,
    vocab_size: int = 25600,
    max_lines: Optional[int] = None,
    special_tokens_num: int = SPECIAL_TOKENS_NUM,
) -> None:
    """训练并保存自定义tokenizer(Qwen风格 + 多模态/工具/think 占位)"""
    os.makedirs(save_dir, exist_ok=True)

    # 初始化tokenizer:ByteLevel 覆盖全部字节,不需要 unk_token 兜底
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = NFKC()  # 文本规范化
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 真·特殊 token:decode 时会被 skip_special_tokens 过滤
    special_tokens_list = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>",
        "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>",
    ]
    # 进入词表但不被视为 special:正常参与解码,不会被 skip_special_tokens 吞掉
    additional_tokens_list = [
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<think>", "</think>",
    ]
    # 预留 buffer 槽位,凑够 special_tokens_num,方便后续扩展不破坏 ID 布局
    num_buffer = special_tokens_num - len(special_tokens_list) - len(additional_tokens_list)
    if num_buffer < 0:
        raise ValueError(
            f"special_tokens_num={special_tokens_num} 小于已配置 token 数 "
            f"{len(special_tokens_list) + len(additional_tokens_list)}"
        )
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)]
    all_reserved_tokens = special_tokens_list + additional_tokens_list + buffer_tokens

    # 把全部预留 token 以 special 身份传入,保证它们占据靠前且连续的 ID
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=all_reserved_tokens,
        min_frequency=5,  # 低频词过滤
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"Training tokenizer with data from {data_path} (max_lines={max_lines})")
    texts = read_texts_from_jsonl(data_path, max_lines=max_lines)
    # length 仅用于进度条估算,子集时按行数给,否则按文件大小
    length = max_lines if max_lines is not None else os.path.getsize(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=length)

    # 验证关键 token 的 ID 顺序
    expected_ids = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2}
    for tok, expected_id in expected_ids.items():
        actual = tokenizer.token_to_id(tok)
        if actual != expected_id:
            raise AssertionError(f"Token {tok} expected id {expected_id}, got {actual}")

    # 用 HF 包装。只把真·特殊 token 中非 bos/eos/pad/unk 的放进 additional_special_tokens
    main_special = {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}
    additional_special = [t for t in special_tokens_list if t not in main_special]

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        additional_special_tokens=additional_special,
        chat_template=CHAT_TEMPLATE,
        model_max_length=131072,
        clean_up_tokenization_spaces=False,
        spaces_between_special_tokens=False,
        add_bos_token=False,
        add_eos_token=False,
        add_prefix_space=False,
        legacy=True,
        # 多模态 token 指针(写入 tokenizer_config.json,供下游多模态代码读取)
        image_token="<|image_pad|>",
        audio_token="<|audio_pad|>",
        video_token="<|video_pad|>",
        vision_bos_token="<|vision_start|>",
        vision_eos_token="<|vision_end|>",
        audio_bos_token="<|audio_start|>",
        audio_eos_token="<|audio_end|>",
    )

    # 一步保存 tokenizer.json + tokenizer_config.json + special_tokens_map.json
    hf_tokenizer.save_pretrained(save_dir)

    # 后处理:把 additional_tokens_list 与 buffer_tokens 的 special 标记翻为 False
    # 这样它们在词表里、可被切分,但 decode(skip_special_tokens=True) 不会过滤它们
    non_special_set = set(additional_tokens_list) | set(buffer_tokens)

    tokenizer_json_path = os.path.join(save_dir, "tokenizer.json")
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
    for tok_info in tokenizer_data.get("added_tokens", []):
        if tok_info["content"] in non_special_set:
            tok_info["special"] = False
    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    config_path = os.path.join(save_dir, "tokenizer_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for tok_info in cfg.get("added_tokens_decoder", {}).values():
        if tok_info["content"] in non_special_set:
            tok_info["special"] = False
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)

    print(f"Tokenizer saved to {save_dir}")

def eval_tokenizer(tokenizer_path: str) -> None:
    """评估tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    text = [
        # 中文样本 (约200字)
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。",
        "星际航行是指在星系内甚至星系间的空间中进行的航行。由于宇宙空间极其广阔，传统的化学火箭动力在恒星间航行时显得力不从心。科学家们提出了多种方案，包括离子推进器、核热火箭、甚至是利用反物质作为能源的设想。此外，曲率驱动和虫洞旅行等科幻概念也在理论物理研究中被反复探讨。尽管目前人类的足迹仅限于月球，但随着核聚变技术和材料科学的突破，前往火星乃至更遥远的太阳系边缘将成为可能。",
        # 英文样本 (约200词/字符)
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information, and the need for significant computational resources.",
        "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply. International cooperation and policy frameworks are also necessary to drive the global shift towards a greener economy and reduce carbon emissions.",
        # 混合样本
        "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science, machine learning, and web development. 开发者可以利用 NumPy, Pandas, and PyTorch 等库快速构建复杂的应用。学习 Python 的过程非常愉快，因为它的代码读起来就像英语一样。Whether you are a beginner or an expert, Python offers something for everyone.",
    ]
    print("\n=== Tokenizer 编码测试 ===")
    enc_ids = tokenizer(text)["input_ids"]
    total_chars = 0
    total_tokens = 0
    for i, (t, ids) in enumerate(zip(text, enc_ids)):
        print(f"[{i}] 字符数 {len(t):>4} → token 数 {len(ids):>4} (压缩比 {len(t)/len(ids):.2f})")
        total_chars += len(t)
        total_tokens += len(ids)
    print(f"[合计] 字符数 {total_chars:>4} → token 数 {total_tokens:>4} "
          f"(平均压缩比 {total_chars/total_tokens:.2f})")

    # 基本属性
    print("\n=== Tokenizer 基本信息 ===")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"BOS / EOS / PAD / UNK: "
          f"{tokenizer.bos_token}({tokenizer.bos_token_id}) / "
          f"{tokenizer.eos_token}({tokenizer.eos_token_id}) / "
          f"{tokenizer.pad_token}({tokenizer.pad_token_id}) / "
          f"{tokenizer.unk_token}({tokenizer.unk_token_id})")
    print(f"All special tokens ({len(tokenizer.all_special_tokens)}): {tokenizer.all_special_tokens}")

    # 多模态 token 指针(从 tokenizer_config.json 的 init_kwargs 读出)
    print("\n=== 多模态 token 指针 ===")
    for k in ["image_token", "audio_token", "video_token",
              "vision_bos_token", "vision_eos_token",
              "audio_bos_token", "audio_eos_token"]:
        v = tokenizer.init_kwargs.get(k)
        tid = tokenizer.convert_tokens_to_ids(v) if v else None
        print(f"  {k}: {v} (id={tid})")

    # 真·特殊 token:skip_special_tokens=True 应被过滤
    print("\n=== 真·特殊 token 过滤 ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    ids = tokenizer(test_text).input_ids
    keep = tokenizer.decode(ids, skip_special_tokens=False)
    skip = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"Original: {test_text}")
    print(f"Decoded(skip=False): {keep}")
    print(f"Decoded(skip=True):  {skip}")
    print(f"  完整 round-trip:        {keep == test_text}")
    print(f"  skip=True 已过滤特殊符: {'<|im_start|>' not in skip and '<|im_end|>' not in skip}")

    # additional/buffer token:整体识别但 skip_special_tokens=True 不过滤(关键特性)
    print("\n=== additional token 处理(think / tool_call) ===")
    test_text = "前缀<think>思考内容</think><tool_call>调用</tool_call>后缀"
    ids = tokenizer(test_text).input_ids
    keep = tokenizer.decode(ids, skip_special_tokens=False)
    skip = tokenizer.decode(ids, skip_special_tokens=True)
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    tool_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    print(f"<think> id={think_id} / <tool_call> id={tool_id}")
    print(f"整体识别: <think> in ids = {think_id in ids}, <tool_call> in ids = {tool_id in ids}")
    print(f"Decoded(skip=True): {skip}")
    print(f"  <think> 在 skip 模式下保留: {'<think>' in skip}")
    print(f"  <tool_call> 在 skip 模式下保留: {'<tool_call>' in skip}")

    # 聊天模板(无 tools)
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]
    print("\n=== 聊天模板(无 tools)===")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print(prompt)

    # 聊天模板(带 tools + add_generation_prompt)
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    print("\n=== 聊天模板(带 tools + add_generation_prompt)===")
    prompt_t = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    print(prompt_t)

    # 编码解码 round-trip
    print("\n=== 编码解码 round-trip ===")
    encoded = tokenizer(prompt)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print(f"Token 数: {len(encoded['input_ids'])}")
    print(f"Round-trip 完全一致: {decoded == prompt}")
    if decoded != prompt:
        print("  (差异多半来自 NFKC 规范化,如全角字符被规范化为半角)")

def main():
    # 以脚本所在目录为基准定位路径,避免依赖 CWD
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    # PROJECT_ROOT = "/root/autodl-tmp/MyLLMDataset"
    data_path = os.path.join(PROJECT_ROOT, "dataset", "mobvoi_seq_monkey_general_open_corpus.jsonl")
    save_dir = os.path.join(PROJECT_ROOT, "tokenizer_k")

    # 训练tokenizer
    # max_lines=None 用全量数据；调试时设成 100000 之类的小数
    # train_tokenizer(
    #     data_path=data_path,
    #     save_dir=save_dir,
    #     vocab_size=6144,
    #     max_lines=None,
    # )
    train_tokenizer(
        data_path=data_path,
        save_dir=save_dir,
        vocab_size=25600,
        max_lines=700000,
    )

    # 评估tokenizer
    eval_tokenizer(save_dir)

if __name__ == '__main__':
    main()