import model_loader
import os
import glob
import pipeline
from PIL import Image
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
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    print(f"Using device: {DEVICE}")

    tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
    model_file = "../data/v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)
    
    ### Input prompts
    prompt = "Cat"
    uncond_prompt = ""
    
    ### Level of guidance
    cfg_scale = 4  # min: 1, max: 14 currently 4 is good

    ### Read Files from input folder
    input_image = None
    input_path = './inputs/'
    output_path = './eval/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    png_files = glob.glob(os.path.join(input_path, '**', '*.png'), recursive=True)
    mask_paths = sorted([file for file in png_files if 'mask' in os.path.basename(file)])
    image_paths = sorted([file for file in png_files if 'mask' not in os.path.basename(file)])

    for image_path, mask_path in zip(image_paths, mask_paths):
        input_image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        sampler = "ddpm"
        num_inference_steps = 50
        seed = 42

        output_image = pipeline.generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=input_image,
            mask=mask,
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