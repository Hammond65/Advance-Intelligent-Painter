import model_loader
import os
import glob
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch

def get_safe_filename(base_name, output_dir):
    index = 0
    while True:
        new_name = f"{base_name}_{index}.png"
        if not os.path.exists(os.path.join(output_dir, new_name)):
            return os.path.join(output_dir, new_name)
        index += 1

if __name__ == "__main__":
    DEVICE = "cpu"

    ALLOW_CUDA = True
    ALLOW_MPS = False

    if torch.cuda.is_available() and ALLOW_CUDA:
        DEVICE = "cuda"
    elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
        DEVICE = "mps"
    print(f"Using device: {DEVICE}")

    tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
    model_file = "../data/v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

    ## TEXT TO IMAGE

    # prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
    prompt = "House near a lake and mountains at the behind, highly detailed, ultra sharp, 100mm lens, 8k resolution."
    # prompt = "" # To Do List - Would it affect resampling
    uncond_prompt = ""  # Also known as negative prompt
    do_cfg = False
    cfg_scale = 8  # min: 1, max: 14

    ## IMAGE TO IMAGE

    input_image = None
    # Comment to disable image to image
    input_path = './inputs/'
    output_path = './outputs/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    png_files = glob.glob(os.path.join(input_path, '**', '*.png'), recursive=True)
    mask_paths = sorted([file for file in png_files if 'mask' in os.path.basename(file)])
    image_paths = sorted([file for file in png_files if 'mask' not in os.path.basename(file)])
    #image_path = "./images/dog.jpg"

    for image_path, mask_path in zip(image_paths, mask_paths):
        input_image = Image.open(image_path).convert('RGB')
        #input_image =None
        mask = Image.open(mask_path).convert('L')
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        # Higher values means more noise will be added to the input image, so the result will further from the input image.
        # Lower values means less noise is added to the input image, so output will be closer to the input image.
        strength = 1.0

        ## SAMPLER

        sampler = "ddpm"
        num_inference_steps = 50
        seed = 42

        output_image = pipeline.generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=input_image,
            mask=mask,
            strength=strength,
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=sampler,
            n_inference_steps=num_inference_steps,
            seed=seed,
            models=models,
            device=DEVICE,
            idle_device="cpu",
            tokenizer=tokenizer,
        )

        # Combine the input image and the output image into a single image.
        img = Image.fromarray(output_image)
        file_name = get_safe_filename(image_name, output_path)
        img.save(file_name)