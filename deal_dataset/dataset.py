import torch
from torch.utils.data import Dataset
from datasets import load_dataset


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