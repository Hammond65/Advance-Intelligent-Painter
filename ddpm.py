import torch
import numpy as np
from torchvision import utils

class DDPMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
        # For the naming conventions, refer to the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator

        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        #timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round().copy().astype(np.int64)
        timesteps = self._set_schedular(num_inference_steps,timesteps)[::-1]
        #self.timesteps = torch.from_numpy(timesteps)
        self.timesteps = torch.tensor(timesteps)
        
    def _set_schedular(self, num_inference_steps, timesteps, jump_length=5, cycle=5, stop_resampling=0):### cycle to 0 to disable loop
        steps_out = []
        
        for idx in range(1, num_inference_steps+1, jump_length):
            if idx == 1:
                steps_out.append(timesteps[0])
            steps_out.extend(timesteps[idx:idx+jump_length])
            if stop_resampling <= timesteps[idx]:
                if idx != range(1, num_inference_steps+1, jump_length)[-1]:
                    for _ in range(cycle):
                        steps_out.extend(timesteps[idx:idx+jump_length-1][::-1])
                        steps_out.extend(timesteps[idx+1:idx+jump_length])
                else:
                    for _ in range(cycle):
                        steps_out.extend(timesteps[idx:idx+jump_length-2][::-1])
                        steps_out.extend(timesteps[idx+1:idx+jump_length])
        steps_out.append(1000)
        #print(steps_out, len(steps_out))
        return steps_out

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        return variance
    
    def set_strength(self, strength=1):
        """
            Set how much noise to add to the input image. 
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    '''def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor, models=None, masked_img=None, gt_mask=None, index: int = None):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        # unrelated = models["decoder"](pred_original_sample)
        # utils.save_image((unrelated+1)/2, f'./pred_x_progress_{index}_{timestep}.png')
        ###Edit
        if gt_mask is not None:
            pred_original_sample = self.masking(timestep, latents, model_output, models, masked_img, gt_mask, index)
            
        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            # Compute the variance as per formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            variance = (self._get_variance(t) ** 0.5) * noise
        
        # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # the variable "variance" is already multiplied by the noise N(0, 1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample'''
        
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor, models=None, masked_img=None, gt_mask=None, index: int = None):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        # unrelated = models["decoder"](pred_original_sample)
        # utils.save_image((unrelated+1)/2, f'./pred_x_progress_{index}_{timestep}.png')
        ###Edit
        if gt_mask is not None:
            pred_original_sample = self.xt_masking(timestep, latents, model_output, models, masked_img, gt_mask, index)
            
        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        mu = pred_original_sample_coeff * pred_original_sample
        mu1 = current_sample_coeff * latents
        #print((1-gt_mask).sum())
        mask_area = gt_mask.sum()
        unmask_area = (1-gt_mask).sum()
        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            # Compute the variance as per formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            variance = (self._get_variance(t) ** 0.5) * noise
        
        # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # the variable "variance" is already multiplied by the noise N(0, 1)
        pred_prev_sample = mu + mu1 + variance

        return pred_prev_sample
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        #print(timesteps)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Sample from q(x_t | x_0) as in equation (4) of https://arxiv.org/pdf/2006.11239.pdf
        # Because N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # here mu = sqrt_alpha_prod * original_samples and sigma = sqrt_one_minus_alpha_prod
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    # def xt_masking(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor, models, encoded_gt, gt_mask, index: int):
    #     t = timestep
    #     device = model_output.device
    #     keep_masked_region = 1-gt_mask
    #     noise = torch.randn(model_output.shape, generator=self.generator, device=device)
        
    #     encoder = models["encoder"].to(device)
    #     decoder = models["decoder"].to(device)
        
    #     # Obtain noisy gt latent
    #     noisy_encoded_gt = self.add_noise(encoded_gt, t)
        
    #     # Obtain the decoded noisy gt and latents
    #     decoded_latents = decoder(latents)
    #     decoded_gt = decoder(noisy_encoded_gt)
        
    #     # Corresponding masked area
    #     masked_gt = decoded_gt * gt_mask
    #     non_masked_region = decoded_latents * gt_mask
        
    #     # Balancing distribution into correct distribution due to masking error causing distribution to shift
    #     encoded_non_masked_region = encoder(non_masked_region, noise)
    #     encoded_masked_gt = encoder(masked_gt, noise)
    #     error_in_non_masked_region = encoded_masked_gt - encoded_non_masked_region
        
    #     err_mean= error_in_non_masked_region.mean()
    #     err_std = error_in_non_masked_region.std()
    #     err_distribution = torch.normal(err_mean, err_std, error_in_non_masked_region.shape, device=device)
    #     shifted_latents = latents + err_distribution
    #     shifted_latents = decoder(shifted_latents)
    #     shifted_masked_region = shifted_latents * keep_masked_region
        
        
        
    #     # Combine masked_gt and shifted_masked_region
    #     img = masked_gt + shifted_masked_region
    #     utils.save_image((img+1)/2, f'./xt_masking_{index}_{timestep}_0.png')
    #     encoded_img = encoder(img, noise)
    #     unrelated = decoder(encoded_img)
    #     utils.save_image((unrelated+1)/2, f'./xt_masking_{index}_{timestep}.png')
    #     return encoded_img
    
    def xt_masking(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor, models, encoded_gt, gt_mask, index: int):
        t = timestep
        device = model_output.device
        keep_masked_region = 1-gt_mask
        noise = torch.randn(model_output.shape, generator=self.generator, device=device)
        
        encoder = models["encoder"].to(device)
        decoder = models["decoder"].to(device)
        
        mask_area = gt_mask.sum()
        not_mask_area = (1-gt_mask).sum()
        # Obtain noisy gt latent
        noisy_encoded_gt = self.add_noise(encoded_gt, t)
        
        # Obtain the decoded noisy gt and latents
        decoded_latents = decoder(latents)
        decoded_gt = decoder(noisy_encoded_gt)
        
        # Corresponding masked area
        masked_gt = decoded_gt * gt_mask
        non_masked_region = decoded_latents * gt_mask
        
        # Balancing distribution into correct distribution due to masking error causing distribution to shift
        encoded_non_masked_region = encoder(non_masked_region, noise)
        encoded_masked_gt = encoder(masked_gt, noise)
        error_in_non_masked_region = (encoded_masked_gt * mask_area - encoded_non_masked_region * not_mask_area) / (mask_area + not_mask_area)
        
        err_mean= error_in_non_masked_region.mean()
        err_std = error_in_non_masked_region.std()
        err_distribution = torch.normal(err_mean, err_std, error_in_non_masked_region.shape, device=device)
        shifted_latents = latents + err_distribution
        shifted_latents = decoder(shifted_latents)
        shifted_masked_region = shifted_latents * keep_masked_region
        
        
        
        # Combine masked_gt and shifted_masked_region
        img = masked_gt + shifted_masked_region
        utils.save_image((img+1)/2, f'./xt_masking_{index}_{timestep}_0.png')
        encoded_img = encoder(img, noise)
        unrelated = decoder(encoded_img)
        utils.save_image((unrelated+1)/2, f'./xt_masking_{index}_{timestep}.png')
        return encoded_img
    
    
    def masking(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor, models, masked_img, gt_mask, index: int):
        t = timestep
        prev_t = self._get_previous_timestep(t)
        device = model_output.device
        keep_masked_region = 1-gt_mask
        noise = torch.randn(model_output.shape, generator=self.generator, device=device)
        
        encoder = models["encoder"].to(device)
        decoder = models["decoder"].to(device)
        
        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        # 2. masking the pred_x0 to our desire pred_x0 with extra information
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        decoded_pred_original_sample = decoder(pred_original_sample)
        modify_decoded_pred_original_sample = decoded_pred_original_sample * keep_masked_region + masked_img * gt_mask
        # utils.save_image((decoded_pred_original_sample+1)/2, './pred_x.png')
        
        latents_pred_x0 = encoder(modify_decoded_pred_original_sample, noise)
        
        # latents = alpha_prod_t.sqrt() * latents_pred_x0 + (1-alpha_prod_t).sqrt() * noise
        # pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        
        # 3. shifting domain
        # shifted_pred_x0 = self._domain_shift(decoded_pred_original_sample, modify_decoded_pred_original_sample, gt_mask)
        # latents_pred_x0 = encoder(shifted_pred_x0, noise)
        
        unrelated = decoder(latents_pred_x0)
        utils.save_image((unrelated+1)/2, f'./pred_x_progress_{index}_{timestep}.png')
        return latents_pred_x0
    
    def undo(self, timestep, latents):
        t = timestep
        prev_t = self._get_previous_timestep(t)
        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        noise = torch.randn(latents.shape, generator=self.generator, device=latents.device, dtype=latents.dtype)
        
        previous_latents = (1-current_beta_t).sqrt() * latents + current_beta_t.sqrt() * noise
        return previous_latents
    
    def _domain_shift(self, original_pred_x0: torch.Tensor, new_pred_x0: torch.Tensor, gt_mask):
        invert_mask = 1 - gt_mask
        
        masked_pred_x0 = original_pred_x0 * gt_mask
        new_masked_pred_x0 = new_pred_x0 * gt_mask
        err_in_pred_x0 = new_masked_pred_x0 - masked_pred_x0
        err_mean = err_in_pred_x0.mean()
        err_std = err_in_pred_x0.std()
        err_distribution = torch.normal(err_mean, err_std, err_in_pred_x0.shape, device=err_in_pred_x0.device)
        
        shifted_pred_x0 = new_pred_x0 * invert_mask + err_distribution * invert_mask + new_pred_x0 * gt_mask
        return shifted_pred_x0