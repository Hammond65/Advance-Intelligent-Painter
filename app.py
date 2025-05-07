import gradio as gr
import torch
import utils.model_loader as model_loader
import utils.pipeline as pipeline
import base64

from PIL import Image
from transformers import CLIPTokenizer
from pathlib import Path
from utils.yaml import load_yaml

example_data_dir = "assets/objects/"
exts = ['jpg', 'jpeg', 'png']
image_files = [p for ext in exts for p in Path(f'{example_data_dir}').glob(f'**/*.{ext}')]
css = """
.center-gallery .gr-gallery {
    display: flex;
    justify-content: center;
    align-items: center;
}
"""

from PIL import Image
import base64
import io

def process_images(base64_str, prompt):
    if not base64_str or len(base64_str) == 0:
        return {"error": "No images provided"}
    
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]

    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    image = image.resize((512, 512))

    print(f"User Prompt: {prompt}")

    alpha = image.getchannel("A")
    mask = alpha.point(lambda a: 255 if a == 255 else 0).convert("L")
    image = image.convert('RGB')
    output_image = pipeline.generate(
            prompt=prompt,
            uncond_prompt=config.uncondition_prompt,
            input_image=image,
            mask=mask,
            cfg_scale=config.cfg_scale,
            sampler_name=config.sampler,
            n_inference_steps=config.num_inference_steps,
            seed=config.seed,
            models=models,
            device=device,
            idle_device="cpu",
            tokenizer=tokenizer,
            config=config,
        )

    return output_image

with open("utils/scripts.js", "r") as js_file:
    js_code = js_file.read()

with open("utils/style.css", "r") as css_file:
    css_code = css_file.read()

html_content = """
<div class="container">
    <div class="centered-content">
        <div class="top-button-group">
            <button class="choose-file-btn" id="chooseFileButton">Choose File</button>
            <button class="action-btn" id="exampleButton">Add Example</button>
        </div>
        <input type="file" id="fileInput" accept="image/*" multiple style="display: none;">
        <canvas id="canvas" class="canvas" width="512" height="512"></canvas>
    </div>
    <div class="button-group">
        <button id="bringToFrontButton" class="action-btn">Bring to Front</button>
        <button id="sendToBackButton" class="action-btn">Send to Back</button>
        <button id="scaleUpButton" class="action-btn">Scale Up</button>
        <button id="scaleDownButton" class="action-btn">Scale Down</button>
        <button id="flipHorizontalButton" class="action-btn">Flip Horizontally</button>
        <button id="processButton" class="process-btn">Process Images</button>
    </div>
</div>
"""

with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")]), css=css_code, js=js_code) as demo:
    gr.HTML('''
            <div style="text-align: center;">
                <h1>Intelligent Picture Painting under Deep Learning with Text Enhancement (Advance-IPainter) 🎨🖌</h1>
            </div>
            </br>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <a href="https://github.com/Hammond65/Advance-Intelligent-Painter" target="_blank">
                    <img src="https://badgen.net/badge/GitHub/AIPainter/blue?icon=github">
                </a>
                &nbsp;
                <a>
                    <img src="https://img.shields.io/badge/Paper-DSP%202025%20-green">
                </a>
                &nbsp;
                <a href="https://www.sfu.edu.hk/en/schools-and-offices/schools-and-departments/school-of-computing-and-information-sciences/introduction/index.html">
                    <img src="https://img.shields.io/badge/CIS-Saint%20\Francis%20University-red">
                </a>
                &nbsp;
                <a href="https://www.polyu.edu.hk/eee/">
                    <img src="https://img.shields.io/badge/EEE-PolyU-yellow">
                </a>
            </div>
            </br>
            ''')
    with gr.Row():
        with gr.Accordion(open=True, label="Source Image"):
            drop_zone = gr.HTML(html_content)
            image_data_input = gr.Textbox(visible=False, label="bridge")
            example_image_input = gr.Image(visible=False, type="pil", elem_id="example_bridge")
            prompt_input = gr.Textbox(label="Enter prompt", placeholder="Describe the image or desired processing...")

        with gr.Accordion(open=True, label="Output Images"):
            gr.Examples(
                examples=[[str(image)] for image in image_files],
                inputs=[example_image_input],
                cache_examples=False,
                examples_per_page=20,
            )
            output = gr.Image(type="pil")
            hidden_button = gr.Button(visible=False, elem_id="button")

    hidden_button.click(fn=process_images, inputs=[image_data_input, prompt_input], outputs=[output])

if __name__ == '__main__':
    config = load_yaml('./config.yaml')
    device = config.device if torch.cuda.is_available() else "cpu"
    print(device)
    
    tokenizer = CLIPTokenizer(config.vocab_path, merges_file=config.merge_path)
    model_file = config.checkpoint
    models = model_loader.preload_models_from_standard_weights(model_file, device)
    
    
    demo.launch()