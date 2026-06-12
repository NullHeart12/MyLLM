import os

import pyarrow.compute as pc

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk

class PretrainDataset(Dataset):
    """
    预训练数据集。包装 HuggingFace `datasets` 的内存映射文件。
    采用懒加载：__init__ 不真正读盘，只记录路径；首次访问时再加载。
    这样在 DataLoader fork 多 worker 时，元数据只在子进程里按需加载。
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self._ds = None

    def _ensure_loaded(self):
        if self._ds is None:
            # 自动识别：目录走 load_from_disk（Arrow，秒级加载）；
            # 单文件走 load_dataset('json',...)（兼容老的 .jsonl 路径）。
            if os.path.isdir(self.data_path):
                self._ds = load_from_disk(self.data_path)
            else:
                self._ds = load_dataset('json', data_files=self.data_path, split='train')

    def __len__(self):
        self._ensure_loaded()
        return len(self._ds)

    def __getitem__(self, idx):
        self._ensure_loaded()
        tokens = self._ds[idx]['input_ids']
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels':    torch.tensor(tokens[1:],  dtype=torch.long),
        }
        
class SFTDataset(Dataset):
    """
    SFT 数据集。包装 HuggingFace `datasets` 的内存映射文件。
    采用懒加载：__init__ 不真正读盘，只记录路径；首次访问时再加载。
    这样在 DataLoader fork 多 worker 时，元数据只在子进程里按需加载。
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self._ds = None

    def _ensure_loaded(self):
        if self._ds is None:
            # 自动识别：目录走 load_from_disk（Arrow 格式，秒级加载）；
            # 单文件走 load_dataset('json',...)（兼容老的 _tokenized.jsonl 路径）。
            if os.path.isdir(self.data_path):
                self._ds = load_from_disk(self.data_path)
            else:
                self._ds = load_dataset('json', data_files=self.data_path, split='train')

    def __len__(self):
        self._ensure_loaded()
        return len(self._ds)

    def __getitem__(self, idx):
        self._ensure_loaded()
        item = self._ds[idx]
        input_ids = item['input_ids']
        labels = item['labels']
        return {
            'input_ids': torch.tensor(input_ids[:-1], dtype=torch.long),
            'labels':    torch.tensor(labels[1:],     dtype=torch.long),
        }
        
    def get_len(self) -> list[int]:
        """返回所有样本的 input_ids 长度，给 LengthGroupedSampler 用。

        用 pyarrow.compute 在 C++ 层向量化算 ListArray 的长度，
        避免逐条 Python 反序列化（3.5M 样本能从几分钟降到 < 1 秒）。
        combine_chunks() 是零拷贝，把多个 chunk 合成单个 Array 方便 pc.list_value_length。
        """
        self._ensure_loaded()
        col = self._ds.data['input_ids'].combine_chunks()
        return pc.list_value_length(col).to_pylist()
        
class SFTCollator:
    """
    SFT 数据集的 Collator。将样本列表拼成 batch。
    这里直接用 tokenizer.pad 来动态 padding，确保 input_ids 和 labels 的 padding 一致。
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch):
        bs = len(batch)
        max_len = max(len(sample['input_ids']) for sample in batch)
        
        input_ids = torch.full((bs, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((bs, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        
        for i, item in enumerate(batch):
            seq_len = len(item['input_ids'])
            input_ids[i, :seq_len] = item['input_ids']
            labels[i, :seq_len] = item['labels']
            attention_mask[i, :seq_len] = 1
            
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
        
class DPODataset(Dataset):
    """
    DPO 数据集。读取 DPOProcessor 产出的 Arrow/JSONL 数据。
    每条样本包含 chosen/rejected 两路 token 与 label，并在 __getitem__ 中做 causal shift。
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self._ds = None

    def _ensure_loaded(self):
        if self._ds is None:
            if os.path.isdir(self.data_path):
                self._ds = load_from_disk(self.data_path)
            else:
                self._ds = load_dataset('json', data_files=self.data_path, split='train')

    def __len__(self):
        self._ensure_loaded()
        return len(self._ds)

    def __getitem__(self, idx):
        self._ensure_loaded()
        item = self._ds[idx]
        chosen_ids = item['chosen_ids']
        chosen_labels = item['chosen_labels']
        rejected_ids = item['rejected_ids']
        rejected_labels = item['rejected_labels']
        return {
            'chosen_ids': torch.tensor(chosen_ids[:-1], dtype=torch.long),
            'chosen_labels': torch.tensor(chosen_labels[1:], dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_ids[:-1], dtype=torch.long),
            'rejected_labels': torch.tensor(rejected_labels[1:], dtype=torch.long),
        }

    def get_len(self) -> list[int]:
        """返回每条样本 chosen/rejected 中较长一路的长度，便于后续按长度分桶。"""
        self._ensure_loaded()
        chosen_col = self._ds.data['chosen_ids'].combine_chunks()
        rejected_col = self._ds.data['rejected_ids'].combine_chunks()
        chosen_lens = pc.list_value_length(chosen_col).to_pylist()
        rejected_lens = pc.list_value_length(rejected_col).to_pylist()
        return [max(c, r) for c, r in zip(chosen_lens, rejected_lens)]

class DPOCollator:
    """
    DPO 数据集的 Collator。chosen/rejected 两路分别动态 padding。
    返回 chosen_attention_mask 和 rejected_attention_mask，供 policy/ref model 前向使用。
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        bs = len(batch)
        max_chosen_len = max(len(sample['chosen_ids']) for sample in batch)
        max_rejected_len = max(len(sample['rejected_ids']) for sample in batch)

        chosen_ids = torch.full((bs, max_chosen_len), self.pad_token_id, dtype=torch.long)
        chosen_labels = torch.full((bs, max_chosen_len), -100, dtype=torch.long)
        chosen_attention_mask = torch.zeros((bs, max_chosen_len), dtype=torch.long)

        rejected_ids = torch.full((bs, max_rejected_len), self.pad_token_id, dtype=torch.long)
        rejected_labels = torch.full((bs, max_rejected_len), -100, dtype=torch.long)
        rejected_attention_mask = torch.zeros((bs, max_rejected_len), dtype=torch.long)

        for i, item in enumerate(batch):
            chosen_len = len(item['chosen_ids'])
            chosen_ids[i, :chosen_len] = item['chosen_ids']
            chosen_labels[i, :chosen_len] = item['chosen_labels']
            chosen_attention_mask[i, :chosen_len] = 1

            rejected_len = len(item['rejected_ids'])
            rejected_ids[i, :rejected_len] = item['rejected_ids']
            rejected_labels[i, :rejected_len] = item['rejected_labels']
            rejected_attention_mask[i, :rejected_len] = 1

        return {
            'chosen_ids': chosen_ids,
            'chosen_labels': chosen_labels,
            'chosen_attention_mask': chosen_attention_mask,
            'rejected_ids': rejected_ids,
            'rejected_labels': rejected_labels,
            'rejected_attention_mask': rejected_attention_mask,
        }
