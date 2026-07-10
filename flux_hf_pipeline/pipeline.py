import torch
import logging
from typing import Dict, List, Union
from enum import Enum
from PIL import Image

from diffusers import FluxPipeline
from ai_pipeline_toolbox.core.pipeline import BaseGenerationPipeline
from ai_pipeline_toolbox.core.models import DynamicModel
from ai_pipeline_toolbox.core.helpers import resolve_air_urn
from ai_pipeline_toolbox.registry.generated_enums import Provider, Category

from flux_hf_pipeline.schemas import FluxConfig, FluxTask

logger = logging.getLogger(__name__)

class Flux1DPipeline(BaseGenerationPipeline[FluxConfig, FluxTask, Image.Image]):
    required_models = []
    
    def setup(self, models_paths: Dict[Union[Enum, str, DynamicModel], str]) -> None:
        logger.info("Initializing Flux Pipeline via HF SDK...")
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        
        # Offload text encoders to CPU immediately to avoid VRAM overload on startup
        self.pipe.text_encoder = self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2 = self.pipe.text_encoder_2.to("cpu")
        
        # Move only Transformer and VAE to GPU
        self.pipe.transformer = self.pipe.transformer.to("cuda")
        self.pipe.vae = self.pipe.vae.to("cuda")
        
        self.models_paths = models_paths
        self.active_lora_urn = None
        self.embeddings_cache = {}
        logger.info("Pipeline initialization complete.")
        
    def get_dynamic_models(self, workload: FluxTask) -> List[DynamicModel]:
        if workload.lora:
            url = resolve_air_urn(workload.lora.urn)
            filename = f"{workload.lora.name}.safetensors"
            return [
                DynamicModel(
                    url=url, 
                    provider=Provider.CIVITAI, 
                    category=Category.LORA.value, 
                    filename=filename
                )
            ]
        return []

    def __call__(self, config: FluxConfig, workload: FluxTask) -> Image.Image:
        prompt = workload.prompt
        
        dynamic_models = self.get_dynamic_models(workload)
        
        if workload.lora and dynamic_models:
            lora_model = dynamic_models[0]
            lora_path = self.models_paths.get(lora_model)
            
            if not lora_path:
                logger.error(f"LoRA path not found for {workload.lora.name}. Is ModelFetcher working?")
                raise FileNotFoundError(f"LoRA path not found in models_paths")
                
            if workload.lora.urn != self.active_lora_urn:
                if self.active_lora_urn is not None:
                    self.pipe.unload_lora_weights()
                    
                logger.info(f"Loading LoRA from {lora_path}")
                self.pipe.load_lora_weights(lora_path)
                self.active_lora_urn = workload.lora.urn
                
            if workload.lora.trigger_words:
                prompt = f"{prompt}, ".join(workload.lora.trigger_words)
        else:
            if self.active_lora_urn is not None:
                self.pipe.unload_lora_weights()
                self.active_lora_urn = None
                
        # Check prompt embeddings cache
        if prompt not in self.embeddings_cache:
            logger.info(f"Cache miss for prompt: '{prompt}'. Encoding text on GPU...")
            # Move text encoders to GPU for fast execution
            self.pipe.text_encoder = self.pipe.text_encoder.to("cuda")
            self.pipe.text_encoder_2 = self.pipe.text_encoder_2.to("cuda")
            
            with torch.no_grad():
                outputs = self.pipe.encode_prompt(prompt=prompt, device="cuda")
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    prompt_embeds = outputs[0]
                    pooled_prompt_embeds = outputs[1]
                else:
                    prompt_embeds = outputs
                    pooled_prompt_embeds = outputs
                    
            # Save to CPU RAM cache
            self.embeddings_cache[prompt] = (prompt_embeds.to("cpu"), pooled_prompt_embeds.to("cpu"))
            
            # Offload text encoders back to CPU to free VRAM
            self.pipe.text_encoder = self.pipe.text_encoder.to("cpu")
            self.pipe.text_encoder_2 = self.pipe.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()
        else:
            logger.info(f"Cache hit for prompt: '{prompt}'. Using cached embeddings.")
            
        # Move embeddings to GPU device
        cached_prompt_embeds, cached_pooled_prompt_embeds = self.embeddings_cache[prompt]
        prompt_embeds = cached_prompt_embeds.to("cuda")
        pooled_prompt_embeds = cached_pooled_prompt_embeds.to("cuda")
                
        logger.info(f"Generating image using cached/loaded embeddings...")
        image = self.pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=config.height,
            width=config.width,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            max_sequence_length=config.max_sequence_length,
            generator=torch.Generator("cuda").manual_seed(config.seed)
        ).images[0]
        
        return image
