import os
import torch
import logging
from typing import Dict, List, Union
from enum import Enum
from PIL import Image

from ccsr import CCSRUpscaler

from ai_pipeline_toolbox.core.pipeline import BaseGenerationPipeline

from ccsr_upscale_pipeline.schemas import PipelineConfig, SingleTask
from ccsr_upscale_pipeline.gdrive_utils import GDriveDownloader

logger = logging.getLogger(__name__)


class CCSRUpscalePipeline(BaseGenerationPipeline[PipelineConfig, SingleTask, dict]):
    required_models = []

    def __init__(self, accelerator=None):
        super().__init__()
        self.accelerator = accelerator

    def setup(self, models_paths: Dict[Union[Enum, str], str]) -> None:
        import ccsr

        ccsr.set_logger(logger)
        logger.info("Setting up CCSRUpscalePipeline using ccsr-pruned...")

        if self.accelerator is not None:
            # Let the main process download and cache the models first
            if self.accelerator.is_main_process:
                logger.info("Main process downloading/caching CCSR models...")
                self._init_upscaler()

            # Wait for the main process to finish downloading and cache population
            self.accelerator.wait_for_everyone()

            # Now all other processes can load models safely from cache
            if not self.accelerator.is_main_process:
                logger.info(
                    f"Process {self.accelerator.process_index} loading CCSR models from cache..."
                )
                self._init_upscaler()
        else:
            self._init_upscaler()

        logger.info("CCSRUpscalePipeline setup complete.")

    def _init_upscaler(self) -> None:
        # Load directly from HuggingFace repository using the wrapper's native from_pretrained() loader
        model_repo = "kharma1/ccsr_v2_repost"

        self.upscaler = CCSRUpscaler(
            controlnet=(model_repo, "controlnet"),
            vae=(model_repo, "vae"),
            unet=(model_repo, "unet", "fp16"),
            text_encoder=(model_repo, "text_encoder", "fp16"),
            tokenizer=(model_repo, "tokenizer"),
            feature_extractor=(model_repo, "feature_extractor"),
            scheduler=(model_repo, "scheduler"),
            sample_method="ddpm",
            mixed_precision="fp16",
            tile_vae=False,
            accelerator=self.accelerator,
        )
        # Enable native diffusers VAE tiling (runs entirely on GPU, lightning-fast)
        self.upscaler.pipeline.vae.tile_sample_min_size = 2048
        self.upscaler.pipeline.vae.tile_latent_min_size = 256
        self.upscaler.pipeline.vae.enable_tiling()

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

        with torch.inference_mode():
            upscaled_image = self.upscaler.upscale(
                image=pil_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                upscale=config.scale_factor,
                tile_diffusion=False,
                seed=config.seed if config.seed != -1 else None,
            )

        results = {
            item.relative_path: {
                "upscale_4k": upscaled_image,
                "item": item,
            }
        }

        return {"dst_root": workload.dst_root, "items": results}
