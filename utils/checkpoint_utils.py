import csv
import torch
import os
from pathlib import Path
import yaml
from omegaconf import OmegaConf
from safetensors.torch import load_file

from models.model import PaletteModel


def load_model_checkpoint(checkpoint_dir, epoch=None):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(
        os.listdir(checkpoint_dir),
        key=lambda x: int(x.split("_")[1].split(".")[0]),
        reverse=True,
    )

    if not checkpoints:
        return None

    if epoch is not None:
        target_name = f"epoch_{epoch}.pth"
        if target_name in checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, target_name)
        else:
            raise FileNotFoundError(
                f"Checkpoint for epoch {epoch} not found in {checkpoint_dir}"
            )
    else:
        checkpoint_path = os.path.join(
            checkpoint_dir, checkpoints[0]
        )  # Load latest checkpoint if epoch not specified
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(checkpoint_path, map_location=device)


def get_last_epoch_from_csv(csv_path):
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if len(rows) == 0:
            return None
        return int(rows[-1]["epoch"])


def load_model_for_inference(config_path, inference_model_path):

    with open(config_path, "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaletteModel(cfg.model).to(device)

    if inference_model_path.endswith(".safetensors"):
        state_dict = load_file(inference_model_path, device=device.type)
        model.load_state_dict(state_dict)
    elif inference_model_path.endswith((".pth", ".pt")):
        obj = torch.load(
            inference_model_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(obj["model"])
    else:
        raise ValueError(f"Unsupported model format: {inference_model_path}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
