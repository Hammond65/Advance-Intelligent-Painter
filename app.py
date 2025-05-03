import gradio as gr
import torch
from PIL import Image
import os.path as osp
from utils.yaml import load_yaml
import time
from transformers import CLIPTokenizer
import utils.model_loader as model_loader
import utils.pipeline as pipeline
import base64
from io import BytesIO
import json

example_data_dir = "assets/"
css = """
.center-gallery .gr-gallery {
    display: flex;
    justify-content: center;
    align-items: center;
}
"""

def process_image_data(data_json):
    data = json.loads(data_json)
    # Example: log the images and positions
    results = []
    for item in data:
        image_data = item['data']
        x, y = item['x'], item['y']
        # Decode image
        image_bytes = base64.b64decode(image_data.split(",")[1])
        image = Image.open(BytesIO(image_bytes))
        results.append(f"Image at ({x}, {y}) with size {image.size}")
    return "\n".join(results)

# Creating the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### Drag-and-Drop Multi-Image Box")

    image_data_json = gr.Textbox(visible=False)
    
    html_box = gr.HTML("""
    <style>
        #drop-box {
            width: 100%;
            height: 400px;
            border: 2px dashed #aaa;
            position: relative;
            overflow: hidden;
        }
        .draggable-img {
            position: absolute;
            max-width: 150px;
            cursor: move;
        }
    </style>
    <div id="drop-box">Drop images here</div>
    <script>
        let dropBox = document.getElementById('drop-box');
        let imageCount = 0;

        dropBox.addEventListener('dragover', e => {
            e.preventDefault();
        });

        dropBox.addEventListener('drop', e => {
            e.preventDefault();
            [...e.dataTransfer.files].forEach(file => {
                if (!file.type.startsWith("image/")) return;

                const reader = new FileReader();
                reader.onload = event => {
                    let img = document.createElement("img");
                    img.src = event.target.result;
                    img.className = "draggable-img";
                    img.style.left = e.offsetX + "px";
                    img.style.top = e.offsetY + "px";
                    dropBox.appendChild(img);
                    makeDraggable(img);
                };
                reader.readAsDataURL(file);
            });
        });

        function makeDraggable(el) {
            let offsetX, offsetY, isDragging = false;
            el.onmousedown = function(e) {
                isDragging = true;
                offsetX = e.offsetX;
                offsetY = e.offsetY;
            };
            document.onmousemove = function(e) {
                if (isDragging) {
                    el.style.left = (e.pageX - dropBox.offsetLeft - offsetX) + 'px';
                    el.style.top = (e.pageY - dropBox.offsetTop - offsetY) + 'px';
                }
            };
            document.onmouseup = function() {
                isDragging = false;
            };
        }

        function collectImageData() {
            const imgs = dropBox.querySelectorAll(".draggable-img");
            const data = [];
            imgs.forEach(img => {
                data.push({
                    data: img.src,
                    x: parseInt(img.style.left),
                    y: parseInt(img.style.top)
                });
            });
            return data;
        }

        // Listen for a custom Gradio event
        window.getImageData = function() {
            const data = collectImageData();
            document.querySelector('textarea').value = JSON.stringify(data);
            document.querySelector('textarea').dispatchEvent(new Event('input', { bubbles: true }));
        }
    </script>
    """)

    process_btn = gr.Button("Process")
    output = gr.Textbox(label="Processed Output", lines=10)

    process_btn.click(fn=None, js="getImageData", inputs=None, outputs=image_data_json)
    image_data_json.change(fn=process_image_data, inputs=image_data_json, outputs=output)

# with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")]), css=css) as demo:
#     gr.HTML('')
#     gr.HTML('''
#             <div style="text-align: center;">
#                 <h1>Intelligent Picture Painting under Deep Learning with Text Enhancement (Advance-IPainter) 🎨🖌</h1>
#             </div>
#             </br>
#             <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
#                 <a href="https://github.com/Hammond65/Advance-Intelligent-Painter" target="_blank">
#                     <img src="https://badgen.net/badge/GitHub/To%20Be%20Announced/blue?icon=github">
#                 </a>
#                 &nbsp;
#                 <a href="https://www.sfu.edu.hk/en/schools-and-offices/schools-and-departments/school-of-computing-and-information-sciences/introduction/index.html">
#                     <img src="https://img.shields.io/badge/CIS-Saint%20\Francis%20University-red">
#                 </a>
#                 &nbsp;
#                 <a href="https://www.polyu.edu.hk/eee/">
#                     <img src="https://img.shields.io/badge/EEE-PolyU-yellow">
#                 </a>
#             </div>
#             </br>
#             ''')
#     with gr.Row():
#         with gr.Accordion(open=True, label="Source Image"):
#             with gr.Row():
#                 # Image upload component
#                 input_image = gr.Image(type="numpy", label="Upload Image", interactive=True)
                
#             process_button = gr.Button("Process Image")
#         with gr.Accordion(open=True, label="Output Images"):
#             output_image = gr.Gallery(label="Prediction", interactive=False, height=300)  # Gallery for showing intermediate images
#     # Resize the uploaded image and send it to the sketchpad
if __name__ == '__main__':
    config = load_yaml('./config.yaml')
    device = config.device if torch.cuda.is_available() else "cpu"
    print(device)
    
    tokenizer = CLIPTokenizer(config.vocab_path, merges_file=config.merge_path)
    model_file = config.checkpoint
    models = model_loader.preload_models_from_standard_weights(model_file, device)
    
    demo.launch()
    # demo.launch(server_port=33333, ssl_verify=False, server_name='0.0.0.0', share=False,
    #         ssl_keyfile="./tobedelete/private.key", ssl_certfile="./tobedelete/certificate.crt")