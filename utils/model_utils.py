import torch


def sample_color_lab(color, noise_std):
    noise = torch.randn_like(color) * noise_std
    sampled = color + noise

    sampled[..., 0].clamp_(0.0, 1.0)
    sampled[..., 1:].clamp_(-1.0, 1.0)
    return sampled
