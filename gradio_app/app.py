import gradio as gr
import json
import torch
import argparse
from huggingface_hub import hf_hub_download

from utils.checkpoint_utils import load_model_for_inference
from utils.color_utils import rgb_to_hex, normalized_lab_to_rgb
from utils.link_utils import github_repo_id, hf_repo_id, hf_model_filename


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to inference model weights (.safetensors / .pth / .pt)",
    )

    return parser.parse_args()


def main(args):
    if args.model is not None:
        model_path = args.model
    else:
        model_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_model_filename)
    model = load_model_for_inference(
        config_path=args.config,
        inference_model_path=model_path,
    )

    def generate(text, palette_size, deterministic):
        html = ""

        with torch.no_grad():
            generated_palette = model.generate(
                text, palette_size=int(palette_size), deterministic=deterministic
            )

        lab = generated_palette[0].float().cpu().numpy()
        hex_palette = [
            rgb_to_hex(normalized_lab_to_rgb(lab_color)) for lab_color in lab
        ]

        html += "<div style='display: flex; flex-direction: row;align-items: center; width:100%;'>"

        hex_codes = []
        for i, hex_color in enumerate(hex_palette):
            hex_color = hex_color.upper()
            hex_codes.append(hex_color)
            html += f'<div style=\'margin:0;flex: 1; text-align: center;\'><div style=\'background-color: {hex_color}; width: 100%; height: 100px;border-radius:{"1em 0 0 1em" if i==0 else "0 1em 1em 0" if i==len(hex_palette)-1 else "0"}\'></div><p style=\'font-size: 14px; margin-top: 5px;\'>{hex_color}</p></div>'
        html += "</div><br>"

        json_output = json.dumps({"palette": hex_codes}, indent=2)
        html += json_output

        return html

    with gr.Blocks() as demo:
        gr.Markdown(
            f"<h1>PaletteLab</h1><br><h4>Generate palettes with text</h4><br>[<a href='https://github.com/{github_repo_id}'>💻Repository</a>]&nbsp;&nbsp;[<a href='https://huggingface.co/spaces/{github_repo_id}'>🕹️Demo</a>]&nbsp;&nbsp;[<a href='https://huggingface.co/{hf_repo_id}'>📦Model</a>]"
        )

        input = gr.Textbox(label="Input text", placeholder="Describe the palette")

        with gr.Row():
            palette_size = gr.Slider(1, 10, value=5, step=1, label="Number of colors")
            deterministic = gr.Checkbox(label="Deterministic", value=False)
        with gr.Row():
            gr.Examples(
                examples=[
                    ["sheep on grassland"],
                    ["peaceful sunset"],
                    ["autumn glow"],
                    ["blueberry milkshake"],
                    ["love and hate"],
                ],
                inputs=[input],
            )

        generate_button = gr.Button("🎨 Generate")
        output = gr.HTML('<div style="height: 100px"></div>')

        generate_button.click(
            generate,
            inputs=[input, palette_size, deterministic],
            outputs=output,
        )
    demo.launch()


if __name__ == "__main__":
    main(parse_args())
