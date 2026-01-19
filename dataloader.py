import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dataset import TextPaletteDataset, palette_collate_fn


def get_dataloaders(
    dataset_path,
    tokenizer,
    tokenizer_input_length,
    test_split=0.2,
    batch_size=32,
    seed=42,
):
    g = torch.Generator().manual_seed(seed)

    dataset = TextPaletteDataset(
        path=dataset_path,
        tokenizer=tokenizer,
        tokenizer_input_length=tokenizer_input_length,
    )

    train_idx, val_idx = train_test_split(
        range(len(dataset)), test_size=test_split, random_state=seed
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=palette_collate_fn,
        generator=g,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=palette_collate_fn
    )

    return train_dataloader, val_dataloader
