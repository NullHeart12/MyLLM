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
