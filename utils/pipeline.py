import torch
import numpy as np
from tqdm import tqdm
from utils.ddpm import DDPMSampler
from PIL import Image
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    mask=None,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
    config=None,
):
    with torch.no_grad():
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        # Prompt Preprocessing
        cond_tokens = tokenizer.batch_encode_plus(
            [prompt], padding="max_length", max_length=77
        ).input_ids
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        cond_context = clip(cond_tokens)
        
        # Unconditional Prompt Preprocessing
        uncond_tokens = tokenizer.batch_encode_plus(
            [uncond_prompt], padding="max_length", max_length=77
        ).input_ids
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
        uncond_context = clip(uncond_tokens)
        
        # Concatenate the conditional and unconditional context
        context = torch.cat([cond_context, uncond_context])

        to_idle(clip)
        del clip
        torch.cuda.empty_cache()
        
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps, config)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # Picture Composition
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            # Image Preprocessing
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Mask Preprocessing
            if mask:
                mask_tensor = preprocess_mask(mask).to(device)
            
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents_gt = encoder(input_image_tensor, encoder_noise)
            latents = torch.randn(latents_shape, generator=generator, device=device)
            to_idle(encoder)
        else:
            # Text to Image generation
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)
        timesteps = tqdm(zip(sampler.timesteps[:-1], sampler.timesteps[1:]), total=(len(sampler.timesteps)-1))
        for i, (t_last, t_cur) in enumerate(timesteps):
            if t_cur < t_last:
                # (1, 320)
                time_embedding = get_time_embedding(t_cur).to(device)
                model_input = latents
                model_input = model_input.repeat(2, 1, 1, 1)
                
                model_output = diffusion(model_input, context, time_embedding)

                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
                latents = sampler.step(t_cur, latents, model_output, models, latents_gt, mask_tensor, i, config)
                
                # Mean Conditional Masking
                if t_cur >= config.mean_masking_stop:
                    latents = sampler.mean_masking(t_cur, latents, model_output, models, latents_gt, mask_tensor)
            else:
                t_last = t_last + sampler.num_train_timesteps // sampler.num_inference_steps
                latents = sampler.undo(t_last, latents)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)
        
        # images = images * (1-tensor_mask(mask).to(device)) + input_image_tensor * tensor_mask(mask).to(device)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def preprocess_mask(mask):
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))
    mask = mask.resize((w // 8, h // 8), resample=Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)
    mask = torch.from_numpy(mask)
    return mask

def tensor_mask(mask):
    mask = mask.convert("L")
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (3, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)
    mask = torch.from_numpy(mask)
    return mask