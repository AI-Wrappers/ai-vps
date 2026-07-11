"""
CCSR (Content Consistent Super-Resolution) upscaling utilities.

Adapted from the official CCSR repository (https://github.com/csslc/CCSR, branch CCSR-v2.0)
to work with diffusers >= 0.39.0 without requiring the full CCSR codebase.

Key logic extracted:
- ControlNet weight loading with condition_conv_in key remapping
- CCSR denoising loop with start_point='lr', t_max/t_min timestep slicing, and initial_step
- Manual ControlNet → UNet forward pass (bypasses standard pipeline's prepare_image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional, Tuple
from PIL import Image
from torchvision.transforms import functional as TF

from diffusers import ControlNetModel, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def load_ccsr_controlnet(cn_path: str, dtype: torch.dtype = torch.bfloat16) -> ControlNetModel:
    """
    Load CCSR ControlNet weights into a standard diffusers ControlNetModel.
    
    CCSR uses `use_vae_encode_condition=True`, meaning the ControlNet expects 
    4-channel VAE latents as conditioning input via a single Conv2d layer called 
    `condition_conv_in` (shape [320, 4, 3, 3]) — applied WITHOUT any activation.
    
    The standard diffusers ControlNetConditioningEmbedding applies SiLU activation 
    between conv_in and conv_out, which distorts the conditioning signal. 
    We replace it with a bare Conv2d to match CCSR's behavior exactly.
    """
    state_dict = load_file(cn_path)
    
    # Create ControlNetModel with latent conditioning config
    controlnet = ControlNetModel(
        in_channels=4,
        conditioning_channels=4,
        conditioning_embedding_out_channels=(320,),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        cross_attention_dim=1024,
        attention_head_dim=8,
        use_linear_projection=True,
    )
    
    # Replace controlnet_cond_embedding with a bare Conv2d (no SiLU activation).
    # CCSR's condition_conv_in is a plain Conv2d(4, 320, 3, padding=1) applied 
    # directly — the standard ControlNetConditioningEmbedding adds SiLU between
    # conv_in and conv_out, which compresses negative values and distorts features.
    bare_conv = nn.Conv2d(4, 320, kernel_size=3, padding=1)
    controlnet.controlnet_cond_embedding = bare_conv
    
    # Remap CCSR keys to the bare conv, skip any original embedding keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k == "condition_conv_in.weight":
            new_state_dict["controlnet_cond_embedding.weight"] = v
        elif k == "condition_conv_in.bias":
            new_state_dict["controlnet_cond_embedding.bias"] = v
        elif k.startswith("controlnet_cond_embedding."):
            # Skip original embedding weights — they were unused during CCSR training
            # (CCSR always uses condition_conv_in when use_vae_encode_condition=True)
            continue
        else:
            new_state_dict[k] = v
    
    missing, unexpected = controlnet.load_state_dict(new_state_dict, strict=False)
    logger.info(f"CCSR ControlNet loaded. Missing keys: {missing}, Unexpected keys: {unexpected}")
    
    return controlnet.to(dtype)


class CCSRUpscaler:
    """
    CCSR upscaling engine that manually drives the denoising loop.
    
    This replaces the CCSR custom StableDiffusionControlNetPipeline with a 
    lightweight wrapper that works with standard diffusers 0.39.0 components.
    
    The key CCSR-specific logic:
    1. Encode the upscaled LR image into VAE latents for ControlNet conditioning
    2. Initialize diffusion latents from the encoded image + noise (start_point='lr')
    3. Perform an initial prediction step to get x0 estimate at t_max
    4. Denoise over the timestep range [t_max, t_min]
    """

    def __init__(self, unet, vae, controlnet, scheduler, text_encoder, tokenizer, device="cuda"):
        self.unet = unet.to(device)
        self.vae = vae.to(device)
        self.controlnet = controlnet.to(device)
        self.scheduler = scheduler
        self.text_encoder = text_encoder.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.image_processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1))
    
    @torch.no_grad()
    def __call__(
        self,
        images: List[Image.Image],
        prompts: List[str],
        height: int,
        width: int,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        t_max: float = 0.6666,
        t_min: float = 0.0,
        start_steps: int = 999,
    ) -> List[Image.Image]:
        """
        Run CCSR upscaling on a batch of PIL images.
        
        Args:
            images: List of PIL images already resized to target (height, width).
            prompts: List of text prompts (can be empty strings).
            height: Target output height.
            width: Target output width.
            num_inference_steps: Number of diffusion steps.
            guidance_scale: CFG scale (1.0 = disabled, recommended for SR).
            t_max: Maximum timestep fraction (0.6666 default from CCSR).
            t_min: Minimum timestep fraction (0.0 default from CCSR).
            start_steps: Starting timestep for noise injection (999 default).
        """
        batch_size = len(images)
        do_cfg = guidance_scale > 1.0
        dtype = self.unet.dtype
        
        # 1. Encode text prompts
        prompt_embeds = self._encode_prompts(prompts, do_cfg)
        
        # 2. Preprocess images to [0, 1] tensor
        image_tensor = self._preprocess_images(images)  # (B, 3, H, W), [0, 1]
        
        # 3. Encode image to VAE latents (CCSR conditioning)
        vae_condition = self.vae.encode(image_tensor * 2 - 1).latent_dist.sample() * self.vae.config.scaling_factor
        
        # 4. Prepare control image for diffusers ControlNet (pass VAE latents directly)
        # We've set up controlnet_cond_embedding to accept 4-channel latent input
        control_cond = vae_condition  # (B, 4, H/8, W/8)
        
        # 5. Set up scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # 6. Prepare initial latents
        latent_shape = (batch_size, 4, height // 8, width // 8)
        latents = torch.randn(latent_shape, device=self.device, dtype=dtype)
        
        # start_point='lr': add noise to the VAE-encoded image
        start_steps_tensor = torch.full((batch_size,), start_steps, device=self.device, dtype=torch.long)
        latents = self.scheduler.add_noise(vae_condition, latents, start_steps_tensor)
        
        # 7. Initial prediction step (predict x0 at start_steps, then re-noise to t_max)
        total_steps = len(timesteps)
        t_tao = timesteps[-round(total_steps * t_max)]
        
        t_init = torch.full((batch_size,), start_steps, device=self.device, dtype=torch.long)
        latent_input = self.scheduler.scale_model_input(latents.to(dtype), t_init)
        
        # Run initial ControlNet + UNet prediction (no CFG for initial step)
        prompt_for_init = prompt_embeds.chunk(2)[0] if do_cfg else prompt_embeds
        control_for_init = control_cond
        
        noise_pred = self._predict_noise(latent_input, t_init, prompt_for_init, control_for_init)
        x0_T = self._predict_x0_from_noise(latents, t_init, noise_pred)
        
        if num_inference_steps == 1:
            latents = x0_T
        else:
            # Re-noise x0 to t_tao
            noise_tao = torch.randn_like(latents)
            latents = self.scheduler.add_noise(x0_T, noise_tao, t_tao)
            
            # 8. Trim timesteps to [t_max, t_min] range
            timesteps = timesteps[-round(total_steps * t_max):]
            if t_min > 0:
                timesteps = timesteps[:-round(total_steps * t_min)]
            
            # 9. Denoising loop
            for t in timesteps:
                latent_input = latents.to(dtype)
                if do_cfg:
                    latent_input = torch.cat([latent_input] * 2)
                latent_input = self.scheduler.scale_model_input(latent_input, t)
                
                cn_prompt = prompt_embeds
                cn_cond = torch.cat([control_cond] * 2) if do_cfg else control_cond
                
                noise_pred = self._predict_noise(latent_input, t, cn_prompt, cn_cond)
                
                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                latents_old = latents
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # Predict x0 for t_min
            if t_min > 0:
                latents = self._predict_x0_from_noise(latents_old, t, noise_pred)
        
        # 10. Decode latents to images
        latents = latents.to(dtype)
        decoded = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0].to(torch.float32)
        images_out = self.image_processor.postprocess(decoded, output_type="pil", do_denormalize=[True] * batch_size)
        
        return images_out
    
    def _encode_prompts(self, prompts: List[str], do_cfg: bool) -> torch.Tensor:
        """Encode text prompts using CLIP text encoder."""
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        if do_cfg:
            uncond_inputs = self.tokenizer(
                [""] * len(prompts),
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_embeds = self.text_encoder(uncond_inputs.input_ids.to(self.device))[0]
            prompt_embeds = torch.cat([negative_embeds, prompt_embeds])
        
        return prompt_embeds
    
    def _preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Convert PIL images to [0, 1] tensor on device."""
        tensors = [TF.to_tensor(img.convert("RGB")) for img in images]
        return torch.stack(tensors).to(device=self.device, dtype=self.unet.dtype)

    def _predict_noise(self, latent_input, t, prompt_embeds, control_cond) -> torch.Tensor:
        """Run ControlNet + UNet to predict noise."""
        down_res, mid_res = self.controlnet(
            latent_input, t,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=control_cond,
            conditioning_scale=1.0,
            return_dict=False,
        )
        noise_pred = self.unet(
            latent_input, t,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
            return_dict=False,
        )[0]
        return noise_pred

    def _predict_x0_from_noise(self, sample, t, noise_pred) -> torch.Tensor:
        """
        Predict the clean image x0 from noisy sample and predicted noise.
        Implements Eq. 15 from the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf).
        """
        t_cpu = t.to(self.scheduler.alphas_cumprod.device)
        if t_cpu.dim() == 0:
            t_cpu = t_cpu.unsqueeze(0)
        t_idx = t_cpu[0].long()
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t_idx].to(sample.device)
        beta_prod_t = 1 - alpha_prod_t
        
        pred_type = self.scheduler.config.prediction_type
        if pred_type == "epsilon":
            return (sample - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        elif pred_type == "sample":
            return noise_pred
        elif pred_type == "v_prediction":
            return alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * noise_pred
        else:
            raise ValueError(f"Unknown prediction_type: {pred_type}")
