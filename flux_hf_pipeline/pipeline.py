import os
import json
import logging
import torch
from typing import Dict, List, Union
from enum import Enum
from PIL import Image

from diffusers import FluxPipeline
from ai_pipeline_toolbox.core.pipeline import BaseGenerationPipeline
from ai_pipeline_toolbox.core.models import DynamicModel
from ai_pipeline_toolbox.core.helpers import resolve_air_urn
from ai_pipeline_toolbox.registry.generated_enums import (
    Provider,
    Category,
    DiT,
    Vae,
    TextEncoders,
)

from flux_hf_pipeline.schemas import FluxConfig, FluxTask

logger = logging.getLogger(__name__)


class Flux1DPipeline(BaseGenerationPipeline[FluxConfig, FluxTask, Image.Image]):
    required_models = [
        DiT.FLUX1D8Q,
        Vae.FLUX1D,
        TextEncoders.CLIP_L,
        TextEncoders.CLIP_T5XXL_FP16,
    ]

    def setup(self, models_paths: Dict[Union[Enum, str, DynamicModel], str]) -> None:
        logger.info("Initializing Flux Pipeline via HF SDK (Fully Offline)...")

        from diffusers import (
            AutoencoderKL,
            FluxTransformer2DModel,
            GGUFQuantizationConfig,
        )
        from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel
        from safetensors.torch import load_file

        transformer_path = str(models_paths[DiT.FLUX1D8Q])
        is_gguf = transformer_path.lower().endswith(".gguf")
        logger.info(
            f"Loading transformer from {transformer_path} (is_gguf: {is_gguf})..."
        )

        transformer = FluxTransformer2DModel.from_single_file(
            transformer_path,
            config="black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
            if is_gguf
            else None,
        )

        logger.info(f"Loading VAE from {models_paths[Vae.FLUX1D]}...")
        vae = AutoencoderKL.from_single_file(
            models_paths[Vae.FLUX1D],
            config="black-forest-labs/FLUX.1-dev",
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )

        # Load CLIP L
        logger.info(f"Loading CLIP from {models_paths[TextEncoders.CLIP_L]}...")
        clip_config = CLIPTextConfig.from_pretrained(
            "black-forest-labs/FLUX.1-dev", subfolder="text_encoder"
        )
        text_encoder = CLIPTextModel(clip_config)
        state_dict = load_file(models_paths[TextEncoders.CLIP_L])
        state_dict = {k.replace("text_model.", ""): v for k, v in state_dict.items()}
        text_encoder.load_state_dict(state_dict, strict=True)
        text_encoder.to(torch.bfloat16)

        # Load T5XXL
        logger.info(f"Loading T5 from {models_paths[TextEncoders.CLIP_T5XXL_FP16]}...")
        t5_config = T5Config.from_pretrained(
            "black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2"
        )
        text_encoder_2 = T5EncoderModel(t5_config)
        state_dict_2 = load_file(models_paths[TextEncoders.CLIP_T5XXL_FP16])
        text_encoder_2.load_state_dict(state_dict_2, strict=False)
        text_encoder_2.to(torch.bfloat16)

        # Instantiate pipeline
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
        )

        self.pipe.to("cuda")

        self.models_paths = models_paths
        self.active_lora_urn = None
        self.first_step_seed = None
        logger.info("Pipeline setup complete.")

    def get_dynamic_models(self, workload: FluxTask) -> List[DynamicModel]:
        if workload.lora:
            url = resolve_air_urn(workload.lora.urn)
            filename = f"{workload.lora.name}.safetensors"
            return [
                DynamicModel(
                    url=url,
                    provider=Provider.CIVITAI,
                    category=Category.LORA.value,
                    filename=filename,
                )
            ]
        return []

    def _resolve_seed(self, seed: int) -> int:
        if seed == -1:
            import random

            run_seed = random.randint(0, 2**32 - 1)
            logger.info(f"Seed set to -1. Generated random seed: {run_seed}")
        elif seed == -2:
            if self.first_step_seed is None:
                import random

                self.first_step_seed = random.randint(0, 2**32 - 1)
                logger.info(
                    f"Seed set to -2 but no first step seed exists. Generated and locked: {self.first_step_seed}"
                )
            run_seed = self.first_step_seed
            logger.info(f"Seed set to -2. Using locked first step seed: {run_seed}")
        elif seed >= 0:
            run_seed = seed
            logger.info(f"Using explicit seed: {run_seed}")
        else:
            run_seed = 0
            logger.info(f"Unsupported negative seed {seed}. Using fallback: {run_seed}")

        if self.first_step_seed is None:
            self.first_step_seed = run_seed
            logger.info(f"First-step seed locked to: {self.first_step_seed}")

        return run_seed

    def __call__(self, config: FluxConfig, workload: FluxTask) -> Image.Image:
        prompt = workload.prompt
        dynamic_models = self.get_dynamic_models(workload)

        target_lora_urn = workload.lora.urn if workload.lora else None
        lora_path = None
        if workload.lora and dynamic_models:
            lora_model = dynamic_models[0]
            lora_path = self.models_paths.get(lora_model)

        # LoRA resolution logic
        if target_lora_urn != self.active_lora_urn:
            logger.info(
                f"LoRA swap triggered. Active: {self.active_lora_urn}, Target: {target_lora_urn}"
            )

            # 1) вивантажити ваги на cpu
            # logger.info("Offloading transformer to CPU...")
            # self.pipe.transformer.to("cpu")
            # torch.cuda.empty_cache()

            # 2) вивантажити лору
            if self.active_lora_urn is not None:
                logger.info("Unloading old LoRA weights...")
                self.pipe.unload_lora_weights()

            # 3) завантажити нову лору
            if workload.lora and lora_path:
                logger.info(f"Loading LoRA weights from {lora_path}...")
                self.pipe.load_lora_weights(lora_path)

            self.active_lora_urn = target_lora_urn

            # 4) завантажити ваги та лору на gpu
            # logger.info(f"Moving transformer back to cuda...")
            # self.pipe.transformer.to("cuda")
        else:
            # If no swap is needed, make sure the transformer is on the target device
            if str(self.pipe.transformer.device) != "cuda":
                logger.info(f"Restoring transformer device to cuda...")
                self.pipe.transformer.to("cuda")

        # Always set strength if LoRA is active and was loaded
        if workload.lora:
            self.pipe.set_adapters(
                ["default"], adapter_weights=[workload.lora.strength]
            )
            if workload.lora.trigger_words:
                prompt = f"{prompt}, " + ", ".join(workload.lora.trigger_words)

        run_seed = self._resolve_seed(config.seed)

        logger.info(f"Generating image for prompt: {prompt} with seed: {run_seed}")

        # Use CPU generator for cross-hardware reproducibility
        generator = torch.Generator(device="cpu").manual_seed(run_seed)

        image = self.pipe(
            prompt,
            height=config.height,
            width=config.width,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            max_sequence_length=config.max_sequence_length,
            generator=generator,
        ).images[0]

        return image
