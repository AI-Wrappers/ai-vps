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

from ccsr_upscale_pipeline.schemas import PipelineConfig, SingleTask
from ccsr_upscale_pipeline.gdrive_utils import GDriveDownloader
from diffusers import AutoencoderKL

logger = logging.getLogger(__name__)

def determine_tile_size(target_width: int, target_height: int, vram_limit_pct: float) -> int:
    if not torch.cuda.is_available():
        return 512

    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_vram_gb = total_memory / (1024**3)
    allowed_vram_gb = total_vram_gb * vram_limit_pct

    weights_footprint_gb = 6.0
    available_vram = allowed_vram_gb - weights_footprint_gb

    if available_vram >= 15.0:
        max_tile = 1024
    elif available_vram >= 9.0:
        max_tile = 768
    elif available_vram >= 5.0:
        max_tile = 512
    else:
        max_tile = 256

    target_max_dim = max(target_width, target_height)
    tile_size = min(max_tile, max(256, target_max_dim))
    
    # Round to nearest multiple of 64
    tile_size = (tile_size // 64) * 64
    logger.info(f"Dynamic VRAM Tiling (CCSR): Calculated Tile Size: {tile_size}")
    return tile_size

class CCSRUpscalePipeline(BaseGenerationPipeline[PipelineConfig, SingleTask, dict]):
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

        # We will always enable tiling for VAE and UNet
        self.pipeline.vae.enable_tiling()
        
        logger.info("Pipeline setup complete.")

    def __call__(self, config: PipelineConfig, workload: SingleTask) -> dict:
        logger.info(f"Executing CCSR upscale task: {workload.task_id}")

        downloader = GDriveDownloader()
        task_idx = 0
        for idx, t in enumerate(downloader.tasks):
            if t.task_id == workload.task_id:
                task_idx = idx
                break

        # Advance sliding window and wait for download to finish
        downloader.on_task_start(task_idx)
        downloader.wait_for_task(workload.task_id)

        item = workload.item
        pil_image = Image.open(item.input_path).convert("RGB")
        prompt = item.prompt
        negative_prompt = item.negative_prompt

        target_width = pil_image.width * config.scale_factor
        target_height = pil_image.height * config.scale_factor
        target_size = (target_width, target_height)

        control_image = pil_image.resize(target_size, Image.Resampling.BICUBIC)
        
        tile_size = determine_tile_size(target_width, target_height, config.vram_limit_pct)
        tile_stride = int(tile_size * config.tile_stride_ratio)

        generator = torch.Generator(device=self.device).manual_seed(config.seed) if config.seed != -1 else None

        with torch.inference_mode():
            upscaled_images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image=control_image,
                num_inference_steps=config.ccsr_steps,
                guidance_scale=config.ccsr_guidance_scale,
                controlnet_conditioning_scale=config.controlnet_conditioning_scale,
                control_guidance_start=config.control_guidance_start,
                control_guidance_end=config.control_guidance_end,
                eta=config.eta,
                color_fix_type=config.color_fix_type,
                generator=generator,
                tile_diffusion=True,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).images

        results = {
            item.relative_path: {
                "upscale_4k": upscaled_images[0],
                "item": item,
            }
        }

        return {"dst_root": workload.dst_root, "items": results}
