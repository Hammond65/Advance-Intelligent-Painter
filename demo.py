import utils.model_loader as model_loader
import os
import glob
import utils.pipeline as pipeline
import torch

from PIL import Image
from transformers import CLIPTokenizer
from utils.yaml import load_yaml
import argparse

def get_safe_filename(base_name, output_dir):
    index = 0
    while True:
        new_name = f"{base_name}_{index}.png"
        if not os.path.exists(os.path.join(output_dir, new_name)):
            return os.path.join(output_dir, new_name)
        index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.yaml", type=str, help="Configuration")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    print(f"Using device: {DEVICE}")

    tokenizer = CLIPTokenizer(config.vocab_path, merges_file=config.merge_path)
    model_file = config.checkpoint
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

    ### Read Files from input folder
    input_image = None
    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)
    png_files = glob.glob(os.path.join(config.input_path, '**', '*.png'), recursive=True)
    mask_paths = sorted([file for file in png_files if 'mask' in os.path.basename(file)])
    image_paths = sorted([file for file in png_files if 'mask' not in os.path.basename(file)])

    for image_path, mask_path in zip(image_paths, mask_paths):
        input_image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        output_image = pipeline.generate(
            prompt=config.prompt,
            uncond_prompt=config.uncondition_prompt,
            input_image=input_image,
            mask=mask,
            cfg_scale=config.cfg_scale,
            sampler_name=config.sampler,
            n_inference_steps=config.num_inference_steps,
            seed=config.seed,
            models=models,
            device=DEVICE,
            idle_device="cpu",
            tokenizer=tokenizer,
        )
        # Combine the input image and the output image into a single image.
        img = Image.fromarray(output_image)
        file_name = get_safe_filename(image_name, config.output_path)
        img.save(file_name)