#!/bin/bash
# 防呆开关:
#   -u  未定义变量就报错,避免 ${empty}/secret 这种意外拼到 /secret 根目录
#   -e  任一命令失败就退出,不再带着错继续跑
set -ue

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 以脚本所在目录为基准定位项目根,数据集统一放在 <项目根>/dataset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
dataset_dir="${PROJECT_ROOT}/dataset"

# 显式确认变量都展开了(防止后续 ${dataset_dir}/xxx 解析成 /xxx 写到根目录)
echo "dataset_dir = ${dataset_dir}"
mkdir -p "${dataset_dir}"

# mkdir -p "${dataset_dir}"

# # 下载预训练数据集， 需要预先安装modelscope，使用pip3 install modelscope安装
# modelscope download \
# --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 \
# --local_dir "${dataset_dir}/seq_monkey"

# # 解压预训练数据集
# tar -xvf "${dataset_dir}/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2" -C "${dataset_dir}"

# 下载SFT数据集
# huggingface-cli download \
#   --repo-type dataset \
#   --resume-download \
#   BelleGroup/train_3.5M_CN \
#   --local-dir "${dataset_dir}/BelleGroup"

# ============================================================
# LoRA continue pretrain 数据集:chinese-poetry GitHub 仓库
# 仓库地址:https://github.com/chinese-poetry/chinese-poetry
# 包含全唐诗 / 全宋词 / 元曲 / 论语 / 诗经 等。本项目只用唐诗 + 宋词。
# ============================================================
# chinese_poetry_target="${dataset_dir}/chinese_poetry"
# chinese_poetry_clone="${dataset_dir}/chinese_poetry_repo"

# if [ -d "${chinese_poetry_target}" ] && [ -n "$(ls -A "${chinese_poetry_target}" 2>/dev/null)" ]; then
#     echo "Skip chinese_poetry: ${chinese_poetry_target} 已存在且非空"
# else
#     mkdir -p "${chinese_poetry_target}"

#     # --depth 1:浅克隆,不要历史,把 ~500MB 压到 ~200MB
#     # GitHub 在国内不稳的话,可以改用 gh-proxy:
#     #   git clone --depth 1 https://gh-proxy.com/https://github.com/chinese-poetry/chinese-poetry.git ...
#     if [ ! -d "${chinese_poetry_clone}" ]; then
#         echo "Clone chinese-poetry 仓库到 ${chinese_poetry_clone}"
#         git clone --depth 1 https://github.com/chinese-poetry/chinese-poetry.git "${chinese_poetry_clone}"
#     fi

#     echo "把需要的 JSON 文件平铺到 ${chinese_poetry_target}"
#     # 全唐诗:poet.tang.*.json + authors.tang.json
#     cp "${chinese_poetry_clone}/全唐诗/"poet.tang.*.json "${chinese_poetry_target}/" 2>/dev/null
#     cp "${chinese_poetry_clone}/全唐诗/"authors.tang.json "${chinese_poetry_target}/" 2>/dev/null

#     # 宋词:ci.song.*.json + author.song.json
#     cp "${chinese_poetry_clone}/宋词/"ci.song.*.json "${chinese_poetry_target}/" 2>/dev/null
#     cp "${chinese_poetry_clone}/宋词/"author.song.json "${chinese_poetry_target}/" 2>/dev/null

#     # 统计 + 可选删除 clone 仓库省磁盘(注释掉默认保留,方便调试)
#     n_files=$(ls "${chinese_poetry_target}"/*.json 2>/dev/null | wc -l)
#     echo "已平铺 ${n_files} 个 JSON 文件到 ${chinese_poetry_target}"
#     # rm -rf "${chinese_poetry_clone}"   # 不需要 clone 仓库了可以取消注释
# fi


# 注意:新版 huggingface_hub (>= 0.25) 已经移除 --resume-download(默认就支持续传),
# 也把命令名从 huggingface-cli 改成了 hf。下面用新写法。
hf download \
  --repo-type dataset \
  ystemsrx/Erotic_Literature_Collection \
  --local-dir "${dataset_dir}/secret"
