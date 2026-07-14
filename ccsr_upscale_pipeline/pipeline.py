import os
import torch
import logging
from typing import Dict, List, Union
from enum import Enum
from PIL import Image

from ccsr import StableDiffusionControlNetCCSRPipeline
from ccsr import ControlNetCCSRModel

from ai_pipeline_toolbox.core.pipeline import BaseGenerationPipeline
from ai_pipeline_toolbox.registry.generated_enums import Checkpoints, Controlnet, Vae

from ccsr_upscale_pipeline.schemas import PipelineConfig, BatchTask
from diffusers import AutoencoderKL

logger = logging.getLogger(__name__)

def determine_batch_size(vram_limit_pct: float) -> int:
    if not torch.cuda.is_available():
        return 1

    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_vram_gb = total_memory / (1024**3)
    allowed_vram_gb = total_vram_gb * vram_limit_pct

    weights_footprint_gb = 6.0
    activation_per_item_gb = 4.5

    available_vram = allowed_vram_gb - weights_footprint_gb
    if available_vram <= 0:
        return 1

    batch_size = max(1, int(available_vram / activation_per_item_gb))
    logger.info(f"Dynamic VRAM Batch Size Calculation (CCSR): Optimal Batch Size: {batch_size}")
    return batch_size

class CCSRUpscalePipeline(BaseGenerationPipeline[PipelineConfig, BatchTask, dict]):
    required_models = [
        Checkpoints.STABLE_DIFFUSION_V2_1,
        Controlnet.CCSR_V2_UPSCALER_CONTROLNET,
        Vae.CCSR_V2_UPSCALER_VAE,
    ]

    def setup(self, models_paths: Dict[Union[Enum, str], str]) -> None:
        logger.info("Setting up CCSRUpscalePipeline components...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        cn_path = models_paths[Controlnet.CCSR_V2_UPSCALER_CONTROLNET]
        vae_path = models_paths[Vae.CCSR_V2_UPSCALER_VAE]
        sd_path = models_paths[Checkpoints.STABLE_DIFFUSION_V2_1]

        logger.info(f"Loading CCSR controlnet from {cn_path}")
        controlnet = ControlNetCCSRModel.from_single_file(cn_path, torch_dtype=torch.bfloat16)

        logger.info(f"Loading CCSR VAE from {vae_path}")
        vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)

        logger.info("Loading CCSR pipeline...")
        self.pipeline = StableDiffusionControlNetCCSRPipeline.from_single_file(
            sd_path,
            config="sd2-community/stable-diffusion-2-1",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.bfloat16
        )
        self.pipeline.to(self.device)

        # Optimization
        self.pipeline.vae.enable_tiling()
        
        logger.info("Pipeline setup complete.")

    def __call__(self, config: PipelineConfig, workload: BatchTask) -> dict:
        logger.info(f"Executing CCSR upscale batch: {workload.task_id}")

        allowed_batch_size = determine_batch_size(config.vram_limit_pct)
        items = workload.items
        
        if len(items) > allowed_batch_size:
            logger.warning(f"Batch size {len(items)} exceeds allowed VRAM safe size {allowed_batch_size}.")

        pil_images = [Image.open(item.input_path) for item in items]
        prompts = [item.prompt for item in items]

        target_dim = 1024 * config.scale_factor
        target_size = (target_dim, target_dim)

        control_images = [
            img.convert("RGB").resize(target_size, Image.Resampling.BICUBIC)
            for img in pil_images
        ]

        with torch.inference_mode():
            upscaled_images = self.pipeline(
                prompt=prompts,
                control_image=control_images,
                num_inference_steps=config.ccsr_steps,
                guidance_scale=config.ccsr_guidance_scale,
                tile_diffusion=True,
                tile_size=512,
            ).images

        results = {}
        for j, item in enumerate(items):
            results[item.relative_path] = {
                "upscale_4k": upscaled_images[j],
                "item": item,
            }

        return {"dst_root": workload.dst_root, "items": results}
