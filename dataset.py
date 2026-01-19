import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json

from utils.color_utils import single_hex_list_to_lab_arr


class TextPaletteDataset(Dataset):
    def __init__(self, path, tokenizer, tokenizer_input_length):
        self.data = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))

        self.tokenizer = tokenizer
        self.tokenizer_input_length = tokenizer_input_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = item["text"]

        palette = single_hex_list_to_lab_arr(item["palette"])
        palette_tensor = torch.tensor(palette, dtype=torch.float32)
        palette_mask = torch.ones(palette_tensor.shape[0], dtype=torch.bool)

        tokens = self.tokenizer(
            text=text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_input_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),  # [T]
            "attention_mask": tokens["attention_mask"].squeeze(0),  # [T]
            "palette": palette_tensor,  # [n_colors, 3]
            "palette_mask": palette_mask,  # [n_colors]
            "text": text,  # string
        }


def palette_collate_fn(batch):

    max_n_colors = max(item["palette"].shape[0] for item in batch)

    input_ids, attention_masks, palettes, palette_masks, texts = [], [], [], [], []

    for item in batch:
        input_ids.append(item["input_ids"])
        attention_masks.append(item["attention_mask"])

        p = item["palette"]
        m = item["palette_mask"]
        pad_len = max_n_colors - p.shape[0]

        if pad_len > 0:
            p = F.pad(p, (0, 0, 0, pad_len))
            m = F.pad(m, (0, pad_len), value=False)

        palettes.append(p)
        palette_masks.append(m)
        texts.append(item["text"])

    padded_batch = {
        "input_ids": torch.stack(input_ids, dim=0),  # [B, T]
        "attention_mask": torch.stack(attention_masks, dim=0),  # [B, T]
        "palette": torch.stack(palettes, dim=0),  # [B, n_colors, 3]
        "palette_mask": torch.stack(palette_masks, dim=0),  # [B, n_colors]
        "text": texts,  # list of strings
    }
    return padded_batch
