# 设置环境变量(仅当前会话生效)
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 以脚本所在目录为基准定位项目根,数据集统一放在 <项目根>/dataset
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$datasetDir = Join-Path $ProjectRoot "dataset"

# 确保目录存在
if (-not (Test-Path $datasetDir)) {
    New-Item -ItemType Directory -Path $datasetDir | Out-Null
}

# 下载预训练数据集,需要预先安装 modelscope: pip install modelscope
modelscope download --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir $datasetDir

# 解压预训练数据集(Windows 下使用 tar,Win10 1803+ 自带)
tar -xvf (Join-Path $datasetDir "mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2") -C $datasetDir

# 下载 SFT 数据集,需要预先安装 huggingface_hub: pip install -U huggingface_hub
huggingface-cli download `
    --repo-type dataset `
    --resume-download `
    BelleGroup/train_3.5M_CN `
    --local-dir (Join-Path $datasetDir "BelleGroup")
