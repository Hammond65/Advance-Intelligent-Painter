import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
from torchvision import utils

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    mask=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

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
        
        if do_cfg:
            # Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            if mask:
                mask_tensor = np.array(mask)
                mask_tensor = mask_tensor.astype(np.float32) / 255.0
                mask_tensor = mask_tensor[None, None]
                mask_tensor[mask_tensor < 0.5] = 0
                mask_tensor[mask_tensor >= 0.5] = 1
                mask_tensor = torch.from_numpy(mask_tensor).to(device)
                print(mask_tensor.shape)
            
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents_gt = encoder(input_image_tensor, encoder_noise)
            
            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)             ### Find input image distribution, mean and variance and add into the empty area
            #latents = sampler.add_noise(latents, sampler.timesteps[0])
            # latents += torch.randn(latents_shape, generator=generator, device=device).mean() ### Edited to add extra noise
            latents = torch.randn(latents_shape, generator=generator, device=device)
            """
            masked_image = input_image_tensor * mask_tensor
            decoded_latent = models["decoder"].to(device)(latents)
            masked_area = decoded_latent * (1-mask_tensor)
            utils.save_image(masked_area, './z/masked_area.png')
            utils.save_image(masked_image, './z/masked_image.png')
            image = masked_image + masked_area
            encode_masked_area = models["encoder"].to(device)(masked_area,encoder_noise)
            encode_masked_image = models["encoder"].to(device)(masked_image,encoder_noise)
            decoded_masked_area = models["decoder"].to(device)(encode_masked_area)
            decoded_masked_image = models["decoder"].to(device)(encode_masked_image)
            utils.save_image(decoded_masked_area, './z/decoded_masked_area.png')
            utils.save_image(decoded_masked_image, './z/decoded_masked_image.png')
            utils.save_image(masked_area, './z/masked_area.png')
            utils.save_image(masked_image, './z/masked_image.png')
            encode_image = encode_masked_area+encode_masked_image
            decode_image = models["decoder"].to(device)(encode_image)
            utils.save_image(decode_image, './z/decode_image.png')
            utils.save_image(image, './z/image.png')
            """

            to_idle(encoder)
        else:
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)
        
        #timesteps = tqdm(sampler.timesteps)
        timesteps = tqdm(zip(sampler.timesteps[:-1], sampler.timesteps[1:]), total=(len(sampler.timesteps)-1))
        for i, (t_last, t_cur) in enumerate(timesteps):
            if t_cur < t_last:
                # (1, 320)
                time_embedding = get_time_embedding(t_cur).to(device)
                
                # (Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = latents

                if do_cfg:
                    # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                    model_input = model_input.repeat(2, 1, 1, 1)

                # model_output is the predicted noise
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
                model_output = diffusion(model_input, context, time_embedding)

                if do_cfg:
                    output_cond, output_uncond = model_output.chunk(2)
                    model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
                # if mask:
                #     sampler.masking(timestep, latents, model_output, models, input_image_tensor, mask_tensor)
                
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
                latents = sampler.step(t_cur, latents, model_output, models, input_image_tensor, mask_tensor, i)
                #print(t_cur)
                # latents = sampler.xt_masking(t_cur, latents, model_output, models, latents_gt, mask_tensor, i)
            else:
                t_last = t_last + sampler.num_train_timesteps // sampler.num_inference_steps
                #print(t_last)
                latents = sampler.undo(t_last, latents)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

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
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
