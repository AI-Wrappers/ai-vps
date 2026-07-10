import torch
import logging
from typing import Dict, List, Union
from enum import Enum
from PIL import Image

from diffusers import FluxPipeline
from ai_pipeline_toolbox.core.pipeline import BaseGenerationPipeline
from ai_pipeline_toolbox.core.models import DynamicModel
from ai_pipeline_toolbox.core.helpers import resolve_air_urn
from ai_pipeline_toolbox.registry.generated_enums import Provider, Category, DiT, Vae, TextEncoders

from flux_hf_pipeline.schemas import FluxConfig, FluxTask

logger = logging.getLogger(__name__)

class Flux1DPipeline(BaseGenerationPipeline[FluxConfig, FluxTask, Image.Image]):
    required_models = [
        DiT.FLUX1D,
        Vae.FLUX1D,
        TextEncoders.CLIP_L,
        TextEncoders.CLIP_T5XXL_FP16
    ]
    
    def setup(self, models_paths: Dict[Union[Enum, str, DynamicModel], str]) -> None:
        logger.info("Initializing Flux Pipeline via HF SDK...")
        
        from diffusers import AutoencoderKL, FluxTransformer2DModel
        from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel
        from torchao.quantization import quantize_, Float8WeightOnlyConfig
        from safetensors.torch import load_file
        
        # Load transformer in BF16 from single file and quantize in-place with torchao
        logger.info(f"Loading transformer from {models_paths[DiT.FLUX1D]}...")
        transformer = FluxTransformer2DModel.from_single_file(
            models_paths[DiT.FLUX1D],
            config="black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            torch_dtype=torch.bfloat16
        )
        # logger.info("Quantizing transformer to float8 via torchao...")
        # quantize_(transformer, Float8WeightOnlyConfig())
        
        # Load VAE from single file
        logger.info(f"Loading VAE from {models_paths[Vae.FLUX1D]}...")
        vae = AutoencoderKL.from_single_file(
            models_paths[Vae.FLUX1D],
            config="black-forest-labs/FLUX.1-dev",
            subfolder="vae",
            torch_dtype=torch.bfloat16
        )
        
        # Load CLIP from single file
        logger.info(f"Loading CLIP from {models_paths[TextEncoders.CLIP_L]}...")
        clip_config = CLIPTextConfig.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder")
        text_encoder = CLIPTextModel(clip_config)
        state_dict = load_file(models_paths[TextEncoders.CLIP_L])
        state_dict = {k.replace("text_model.", ""): v for k, v in state_dict.items()}
        text_encoder.load_state_dict(state_dict, strict=True)
        text_encoder.to(torch.bfloat16)
        
        # Load T5 from single file (strict=False to ignore tied embedding parameter warning)
        logger.info(f"Loading T5 from {models_paths[TextEncoders.CLIP_T5XXL_FP16]}...")
        t5_config = T5Config.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2")
        text_encoder_2 = T5EncoderModel(t5_config)
        state_dict_2 = load_file(models_paths[TextEncoders.CLIP_T5XXL_FP16])
        text_encoder_2.load_state_dict(state_dict_2, strict=False)
        text_encoder_2.to(torch.bfloat16)
        
        # Initialize pipeline with pre-loaded components
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16
        )
        # self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()
        
        self.models_paths = models_paths
        self.active_lora_urn = None
        self.generator = torch.Generator("cpu")
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
                    
                logger.info(f"Loading LoRA from {lora_path}...")
                self.pipe.load_lora_weights(lora_path)
                self.active_lora_urn = workload.lora.urn
                
            if workload.lora.trigger_words:
                prompt = f"{prompt}, ".join(workload.lora.trigger_words)
        elif self.active_lora_urn is not None:
            self.pipe.unload_lora_weights()
            self.active_lora_urn = None
            
        # Re-create the scheduler with ComfyUI's dynamic timestep shift
        from diffusers import FlowMatchEulerDiscreteScheduler
        x1 = 256
        x2 = 4096
        mm = (1.15 - 0.5) / (x2 - x1)
        b = 0.5 - mm * x1
        shift = (config.width * config.height / 256) * mm + b
        
        logger.info(f"Setting scheduler timestep shift to {shift:.4f} (matching ComfyUI)")
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            shift=shift
        )
                 
        logger.info(f"Generating image for prompt: {prompt}")
        image = self.pipe(
            prompt,
            height=config.height,
            width=config.width,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            max_sequence_length=config.max_sequence_length,
            generator=self.generator.manual_seed(config.seed)
        ).images[0]
        
        return image
