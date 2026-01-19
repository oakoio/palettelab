import torch
import torch.nn as nn
from torch import nn, Tensor
from transformers import CLIPTextModel, CLIPTokenizer
import math

from utils.model_utils import sample_color_lab

# B - batch size
# S - palette sequence length
# L - text token length
# D - dimension


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, positions: Tensor) -> Tensor:
        assert positions.max().item() < self.pe.size(0), "Sequence exceeds maximum length for positional encoding"  # type: ignore
        x = self.pe[positions]  # type: ignore
        return self.dropout(x)


class PaletteModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        clip_model_name = cfg.clip_model_name
        tokenizer_input_length = cfg.tokenizer_input_length
        d_text_proj = cfg.d_text_proj
        d_model = cfg.d_model
        self.d_model = d_model
        d_z = cfg.d_z
        self.d_z = d_z
        n_layers = cfg.n_layers
        n_heads = cfg.n_heads
        dim_ff = cfg.dim_ff

        self.max_seq_len = cfg.max_seq_len
        self.teacher_forcing_noise = cfg.teacher_forcing_noise

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.clip_text = CLIPTextModel.from_pretrained(clip_model_name)

        # freeze CLIP except the last transformer block
        for name, p in self.clip_text.named_parameters():
            if "layer.11" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.tokenizer_input_length = tokenizer_input_length

        self.text_proj = nn.Sequential(
            nn.Linear(d_text_proj, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.2),
        )

        self.color_embed = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Dropout(0.2)
        )
        self.start_embed = nn.Parameter(torch.randn(1, d_model))
        self.position_embed = PositionalEncoding(
            d_model=d_model, max_len=self.max_seq_len
        )
        self.z_proj = nn.Sequential(
            nn.Linear(d_z, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.2),
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, n_heads, dim_ff, dropout=0.2, batch_first=True
            ),
            num_layers=n_layers,
        )

        self.head_l = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.head_ab = nn.Sequential(nn.Linear(d_model, 2), nn.Tanh())

    def forward(
        self,
        input_ids,
        attention_mask,
        palette,
        palette_mask,
        random_noise_conditioning=True,
    ):
        device = input_ids.device

        B, S, _ = palette.shape  # [B, S, 3]

        text_feats = self.clip_text(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        text_feats = self.text_proj(text_feats)  # [B, L, D]

        # Noisy teacher forcing during training
        palette_trunc = palette[:, :-1]  # never sees the last color
        if self.training:
            palette_trunc = (
                palette_trunc
                + torch.randn_like(palette_trunc) * self.teacher_forcing_noise
            )

        decoder_in = self.color_embed(palette_trunc)  # [B, S-1, D]
        bos = self.start_embed.expand(B, 1, -1)
        decoder_in = torch.cat([bos, decoder_in], dim=1)  # [B, S, D]

        position_emb = self.position_embed(torch.arange(S, device=device)).unsqueeze(
            0
        )  # [1, S, D]
        decoder_in = decoder_in + position_emb

        # inject noise conditioning
        if random_noise_conditioning:
            z = torch.randn(B, self.d_z, device=device)
        else:
            z = torch.zeros(B, self.d_z, device=device)
        z_proj = self.z_proj(z).unsqueeze(1)  # [B, 1, D]
        decoder_in = decoder_in + z_proj  # [B, S, D]

        tgt_mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1
        )
        tgt_key_padding_mask = torch.cat(
            [
                torch.zeros(
                    B, 1, device=device, dtype=torch.bool
                ),  # BOS is always valid
                ~palette_mask[
                    :, :-1
                ],  # palette_mask: True for valid, False for invalid
            ],
            dim=1,
        )
        output = self.decoder(
            decoder_in,
            text_feats,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=~attention_mask.bool(),
        )

        out_l = self.head_l(output)
        out_ab = self.head_ab(output)
        out = torch.cat([out_l, out_ab], dim=-1)

        return out  # [B, S, 3]

    @torch.no_grad()
    def generate(
        self,
        text,
        palette_size,
        deterministic=False,
        stochastic_output_noise_std=0.05,
    ):
        assert (
            palette_size <= self.max_seq_len
        ), "Requested palette size exceeds model's maximum sequence length"
        self.eval()
        device = next(self.parameters()).device

        B = 1

        tokenized = self.tokenizer(
            text,
            max_length=self.tokenizer_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        text_feats = self.clip_text(**tokenized).last_hidden_state
        text_feats = self.text_proj(text_feats)  # [B, L, D]

        current_emb = self.start_embed.expand(B, 1, -1)  # [1, 1, D]
        colors = []

        if not deterministic:
            z = torch.randn(B, self.d_z, device=device)
        else:
            z = torch.zeros(B, self.d_z, device=device)
        z_proj = self.z_proj(z).unsqueeze(1)  # [B, 1, D]

        for i in range(palette_size):
            S = i + 1
            position_emb = self.position_embed(
                torch.arange(0, S, device=device)
            ).unsqueeze(
                0
            )  # [1, i+1, D]

            decoder_in = current_emb + position_emb

            # inject noise conditioning
            decoder_in = decoder_in + z_proj  # [1, i+1, D]

            tgt_mask = torch.triu(
                torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1
            )

            output = self.decoder(
                decoder_in,
                text_feats,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=~tokenized["attention_mask"].bool(),
            )  # [1, i+1, D]

            out_l = self.head_l(output)
            out_ab = self.head_ab(output)
            out = torch.cat([out_l, out_ab], dim=-1)

            out_last = out[:, -1:, :]  # [1, 1, D]
            if not deterministic:
                out_last = sample_color_lab(
                    out_last, noise_std=stochastic_output_noise_std
                )

            color_emb = self.color_embed(out_last)  # [1, 1, D]
            current_emb = torch.cat([current_emb, color_emb], dim=1)  # [1, i+2, D]
            colors.append(out_last)
        return torch.cat(colors, dim=1)
