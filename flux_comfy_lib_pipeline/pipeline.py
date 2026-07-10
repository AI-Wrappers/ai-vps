import logging
from typing import Dict, List, Union
from enum import Enum
from PIL import Image
from pathlib import Path

from ai_pipeline_toolbox.core.pipeline import BaseGenerationPipeline
from ai_pipeline_toolbox.core.models import DynamicModel
from ai_pipeline_toolbox.registry.generated_enums import DiT, Vae, TextEncoders

from flux_hf_pipeline.schemas import FluxConfig, FluxTask

logger = logging.getLogger(__name__)

class FluxComfyLibPipeline(BaseGenerationPipeline[FluxConfig, FluxTask, Image.Image]):
    required_models = [
        DiT.FLUX1D,
        Vae.FLUX1D,
        TextEncoders.CLIP_L,
        TextEncoders.CLIP_T5XXL_FP16
    ]
    
    def setup(self, models_paths: Dict[Union[Enum, str, DynamicModel], str]) -> None:
        logger.info("Initializing Flux Pipeline via ComfyUI libraries...")
        
        import sys
        # Ensure ComfyUI path is at the front of python search path
        sys.path.insert(0, "/workspace/ComfyUI")
        
        # Bypass default ComfyUI command line parsing
        import comfy.options
        comfy.options.args_parsing = False
        
        import comfy.sd
        import comfy.cli_args
        
        # Enable High VRAM mode for local execution
        comfy.cli_args.args.highvram = True
        comfy.cli_args.args.gpu_only = False
        
        self.models_paths = models_paths.copy()
        
        # Load the models via ComfyUI libraries
        logger.info(f"Loading DiT Transformer from {models_paths[DiT.FLUX1D]}...")
        self.model = comfy.sd.load_diffusion_model(models_paths[DiT.FLUX1D])
        
        logger.info("Loading Dual CLIP Text Encoders...")
        clip_l_path = models_paths[TextEncoders.CLIP_L]
        t5_path = models_paths[TextEncoders.CLIP_T5XXL_FP16]
        self.clip = comfy.sd.load_clip(
            ckpt_paths=[clip_l_path, t5_path],
            clip_type=comfy.sd.CLIPType.FLUX
        )
        
        logger.info(f"Loading VAE from {models_paths[Vae.FLUX1D]}...")
        import comfy.utils
        vae_sd, vae_metadata = comfy.utils.load_torch_file(models_paths[Vae.FLUX1D], return_metadata=True)
        self.vae = comfy.sd.VAE(sd=vae_sd, metadata=vae_metadata)
        
        # Caching patched model and clip to avoid rebuilding them on every execution
        self.cached_patched_model = None
        self.cached_patched_clip = None
        self.cached_lora_urn = None
        self.cached_lora_strength = None
        self.cached_width = None
        self.cached_height = None
        
    def get_dynamic_models(self, workload: FluxTask) -> List[DynamicModel]:
        if workload.lora:
            return [DynamicModel(
                name=workload.lora.name,
                urn=workload.lora.urn,
                url=workload.lora.url,
                category="Lora"
            )]
        return []
        
    def __call__(self, config: FluxConfig, workload: FluxTask) -> Image.Image:
        logger.info(f"Executing ComfyUI Library Pipeline for workload: {workload.task_id}")
        
        import torch
        import comfy.sd
        import comfy.utils
        import comfy.sample
        import comfy.model_management
        import comfy_extras.nodes_model_advanced
        import nodes
        import time
        import numpy as np
        
        # 1. Resolve and apply LoRA and patch model sampling (with caching)
        lora_urn = workload.lora.urn if workload.lora else None
        lora_strength = workload.lora.strength if workload.lora else None
        
        if (
            self.cached_patched_model is not None 
            and self.cached_lora_urn == lora_urn 
            and self.cached_lora_strength == lora_strength
            and self.cached_width == config.width
            and self.cached_height == config.height
        ):
            logger.info("Reusing cached patched model and CLIP...")
            patched_model = self.cached_patched_model
            clip = self.cached_patched_clip
        else:
            logger.info("No cache hit. Rebuilding patched model and CLIP...")
            model = self.model
            clip = self.clip
            
            if workload.lora:
                lora_model = self.get_dynamic_models(workload)[0]
                lora_path = self.models_paths.get(lora_model) or self.models_paths.get(workload.lora.urn)
                if not lora_path:
                    for k, v in self.models_paths.items():
                        if getattr(k, 'urn', None) == workload.lora.urn or str(k) == workload.lora.urn:
                            lora_path = v
                            break
                
                if not lora_path:
                    logger.error(f"LoRA path not found in models_paths for URN: {workload.lora.urn}")
                    raise FileNotFoundError(f"LoRA path not found in models_paths")
                    
                # If the clean name version exists in the cache folder, prefer it
                lora_filename = f"{workload.lora.name}.safetensors"
                clean_path = Path(lora_path).parent / lora_filename
                if clean_path.exists():
                    lora_path = str(clean_path)
                    
                logger.info(f"Applying LoRA: {workload.lora.name} (strength: {workload.lora.strength})")
                # Load LoRA weights
                lora_weights, lora_metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
                # Patch model and clip
                model, clip = comfy.sd.load_lora_for_models(
                    model, 
                    clip, 
                    lora_weights, 
                    workload.lora.strength, 
                    workload.lora.strength, 
                    lora_metadata=lora_metadata
                )
                
            # 2. Patch Model Sampling for Flux shift
            logger.info("Patching Model Sampling for Flux...")
            ms_flux = comfy_extras.nodes_model_advanced.ModelSamplingFlux()
            patched_model = ms_flux.patch(
                model, 
                max_shift=1.15, 
                base_shift=0.5, 
                width=config.width, 
                height=config.height
            )[0]
            
            # Update cache
            self.cached_patched_model = patched_model
            self.cached_patched_clip = clip
            self.cached_lora_urn = lora_urn
            self.cached_lora_strength = lora_strength
            self.cached_width = config.width
            self.cached_height = config.height
        
        # 3. CLIP Text Encoding for Flux
        logger.info("Encoding prompt...")
        tokens = clip.tokenize(workload.prompt)
        t5_tokens = clip.tokenize(workload.prompt)
        if "t5xxl" in t5_tokens:
            tokens["t5xxl"] = t5_tokens["t5xxl"]
            
        cond = clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": config.guidance_scale})
        
        # 4. Create empty latent space (16 channels, downscaled by 8)
        latent_tensor = torch.zeros(
            [1, 16, config.height // 8, config.width // 8], 
            device=comfy.model_management.intermediate_device(), 
            dtype=comfy.model_management.intermediate_dtype()
        )
        latent = {"samples": latent_tensor}
        
        # 5. Execute Sampler via common_ksampler helper
        logger.info("Running Sampler...")
        start_time = time.time()
        
        sampled_latent = nodes.common_ksampler(
            model=patched_model,
            seed=config.seed,
            steps=config.num_inference_steps,
            cfg=1.0, # CFG is 1.0 for Flux guidance-based workflows
            sampler_name="euler",
            scheduler="normal",
            positive=cond,
            negative=[],
            latent=latent,
            denoise=1.0
        )[0]
        
        elapsed = time.time() - start_time
        logger.info(f"Sampling completed in {elapsed:.2f} seconds.")
        
        # 6. Decode latent space via VAE
        logger.info("Decoding latent space via VAE...")
        decoded = self.vae.decode(sampled_latent["samples"])
        
        # Convert PyTorch tensor to PIL Image (channel format is [B, H, W, C])
        decoded_tensor = decoded[0].cpu().numpy()
        img_np = np.clip(255.0 * decoded_tensor, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        return img
