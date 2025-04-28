import gradio as gr
import torch
from PIL import Image
import os.path as osp
import cv2
import numpy as np
import time
example_data_dir = "assets/"
css = """
.center-gallery .gr-gallery {
    display: flex;
    justify-content: center;
    align-items: center;
}
"""
# Function to resize the image to 256x256
def align_face(image, desired_size=512):
    return image

def loop_images(images):
    """Loop through images and update the gallery."""
    for img in images:
        yield [img]  # Output must be wrapped in a list for Gallery
        time.sleep(0.5)  # Adjust speed as needed
        
# Function to process the edited image and mask
def process_input(ims):
    input_image = Image.fromarray(ims["composite"])
    width, height = input_image.size
    if width % 16 != 0:
        width -= width % 16
        input_image = input_image.crop((0, 0, width, height))
    if height % 16 != 0:
        height -= height % 16
        input_image = input_image.crop((0, 0, width, height))
    mask = 255-ims["layers"][0][:,:,3]
    mask[mask < 128] = 0
    mask[mask >= 128] = 255
    mask = Image.fromarray(mask)
    mask = mask.crop((0, 0, input_image.width, input_image.height))
    print(mask.size, input_image.size)
    output, intermediates = trainer.test(input_image, mask)
    intermediates.append(output)
    return intermediates
# Creating the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")]), css=css) as demo:
    gr.HTML('')
    gr.HTML('''
            <div style="text-align: center;">
                <h1>Intelligent Picture Painting under Deep Learning with Text Enhancement (Advance-IPainter) 🎨🖌</h1>
            </div>
            </br>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <a href="https://github.com/Hammond65/DTLS" target="_blank">
                    <img src="https://badgen.net/badge/GitHub/To%20Be%20Announced/blue?icon=github">
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
            with gr.Row():
                # Image upload component
                input_image = gr.Image(type="numpy", label="Upload Image", interactive=True)
                # Editable image for inpainting (now using Sketchpad for drawing)
                paint_image = gr.ImageMask(type="numpy", layers=True, label="Draw Box",brush=gr.Brush(color_mode='fixed',default_size=5))
            # Button to trigger processing
            gr.Examples(
                    examples=[
                        [osp.join(example_data_dir, "image0008.jpg")],
                        [osp.join(example_data_dir, "image0011.jpg")],
                        [osp.join(example_data_dir, "image0027.jpg")],
                        [osp.join(example_data_dir, "image0042.jpg")],
                        [osp.join(example_data_dir, "images.jpg")],
                    ],
                    inputs=[input_image],
                    cache_examples=False,
                    examples_per_page=20,
                )
            process_button = gr.Button("Process Image")


        with gr.Accordion(open=True, label="Output Images"):
            output_image = gr.Gallery(label="Prediction", interactive=False, height=300)  # Gallery for showing intermediate images
    # Resize the uploaded image and send it to the sketchpad
    input_image.change(fn=align_face, inputs=input_image, outputs=paint_image)
    # Process image only when the button is clicked
    process_button.click(fn=process_input, inputs=[paint_image], outputs=output_image) #.then(fn=loop_images, inputs=[output_image], outputs=[output_image])
if __name__ == '__main__':
    config = load_yaml('./config/trainer/appv2.yaml')
    device = config.device if torch.cuda.is_available() else "cpu"
    print(device)
    model = model(**config.networks.refine_unet).to(device)

    dtls = dtls(
        model,
        **config.networks.refine_dtls,
        device=device,
    ).to(device)

    trainer = trainer(
        dtls,
        None,
        None,
        **config.trainer,
        device=device,
    )
    
    demo.launch()
    # demo.launch(server_port=33333, ssl_verify=False, server_name='0.0.0.0', share=False,
    #         ssl_keyfile="./tobedelete/private.key", ssl_certfile="./tobedelete/certificate.crt")