import os
import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Union
from enum import Enum
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from transformers import AutoConfig, AutoModelForImageSegmentation
from safetensors.torch import load_file

from diffusers import StableDiffusionControlNetPipeline, AutoencoderKL

from upscale_rmbg_pipeline.ccsr_utils import load_ccsr_controlnet, CCSRUpscaler

from ai_pipeline_toolbox.core.pipeline import BaseGenerationPipeline
from ai_pipeline_toolbox.registry.generated_enums import Checkpoints, Controlnet, Vae, RmbgModels

from upscale_rmbg_pipeline.schemas import PipelineConfig, BatchTask

logger = logging.getLogger(__name__)

def guided_filter_torch(guide: torch.Tensor, src: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """
    Applies Guided Filter to refine src mask guided by high-res guide.
    Operates entirely in PyTorch on GPU using replication padding for border quality.
    
    guide: torch.Tensor of shape (B, C, H, W)
    src: torch.Tensor of shape (B, 1, H, W)
    r: radius
    eps: epsilon
    """
    box_size = 2 * r + 1
    
    # Convert guide to grayscale
    if guide.shape[1] == 3:
        guide_gray = (
            0.299 * guide[:, 0:1] + 0.587 * guide[:, 1:2] + 0.114 * guide[:, 2:3]
        )
    else:
        guide_gray = guide

    # Replication padding avoids border artifacts (dark borders)
    padding = (r, r, r, r)
    
    def box_filter(x):
        x_padded = F.pad(x, padding, mode="replicate")
        return F.avg_pool2d(x_padded, kernel_size=box_size, stride=1, padding=0)
        
    mean_I = box_filter(guide_gray)
    mean_p = box_filter(src)
    mean_Ip = box_filter(guide_gray * src)
    
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = box_filter(guide_gray * guide_gray)
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = box_filter(a)
    mean_b = box_filter(b)
    
    q = mean_a * guide_gray + mean_b
    return q

def determine_batch_size(vram_limit_pct: float) -> int:
    """
    Dynamically determines optimal batch size based on physical GPU VRAM and user allowance.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Defaulting batch size to 1.")
        return 1

    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_vram_gb = total_memory / (1024**3)
    allowed_vram_gb = total_vram_gb * vram_limit_pct

    # Model Weights Footprint:
    # RMBG + SD 2.1 + CCSR CN + CCSR VAE = ~8 GB (due to precision buffers and PyTorch CUDA context)
    weights_footprint_gb = 8.0

    # Activation Memory Footprint per 4k upscale item:
    # SD 2.1 ControlNet + Tiled VAE = ~4.5 GB
    activation_per_item_gb = 4.5

    available_vram = allowed_vram_gb - weights_footprint_gb
    if available_vram <= 0:
        logger.warning(
            f"Allowed VRAM ({allowed_vram_gb:.2f} GB) is less than model weights requirement. Defaulting batch size to 1."
        )
        return 1

    batch_size = max(1, int(available_vram / activation_per_item_gb))
    logger.info(f"Dynamic VRAM Batch Size Calculation:")
    logger.info(f"  Total GPU VRAM: {total_vram_gb:.2f} GB")
    logger.info(
        f"  Allowed VRAM ({vram_limit_pct * 100:.1f}%): {allowed_vram_gb:.2f} GB"
    )
    logger.info(f"  Estimated Weights: {weights_footprint_gb} GB")
    logger.info(f"  Optimal Batch Size: {batch_size}")

    return batch_size

class BgRemovalUpscalePipeline(BaseGenerationPipeline[PipelineConfig, BatchTask, dict]):
    required_models = [
        Checkpoints.STABLE_DIFFUSION_V2_1,
        Controlnet.CCSR_V2_UPSCALER_CONTROLNET,
        Vae.CCSR_V2_UPSCALER_VAE,
        RmbgModels.BRIIA_RMBG_V2,
    ]

    def setup(self, models_paths: Dict[Union[Enum, str], str]) -> None:
        logger.info("Setting up BgRemovalUpscalePipeline components...")
        self.device = "cuda"

        # 1. Load RMBG-2.0
        rmbg_path = models_paths[RmbgModels.BRIIA_RMBG_V2]
        logger.info(f"Loading RMBG-2.0 weights from {rmbg_path}...")
        # AutoModelForImageSegmentation expects BriaAI custom architecture definition
        config = AutoConfig.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)
        self.rmbg = AutoModelForImageSegmentation.from_config(
            config,
            trust_remote_code=True,
        )
        self.rmbg.load_state_dict(load_file(rmbg_path))
        self.rmbg.to(self.device).eval()

        # 2. Load CCSR ControlNet
        cn_path = models_paths[Controlnet.CCSR_V2_UPSCALER_CONTROLNET]
        logger.info(f"Loading CCSR ControlNet from {cn_path}...")
        controlnet = load_ccsr_controlnet(cn_path, dtype=torch.bfloat16)

        # 3. Load CCSR VAE
        vae_path = models_paths[Vae.CCSR_V2_UPSCALER_VAE]
        logger.info(f"Loading CCSR VAE from {vae_path}...")
        vae = AutoencoderKL.from_single_file(
            vae_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )

        # 4. Load Base SD 2.1 components (UNet, Text Encoder, Tokenizer, Scheduler)
        sd_path = models_paths[Checkpoints.STABLE_DIFFUSION_V2_1]
        logger.info(f"Loading Stable Diffusion 2.1 from {sd_path}...")
        temp_pipe = StableDiffusionControlNetPipeline.from_single_file(
            sd_path,
            config="sd2-community/stable-diffusion-2-1",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.bfloat16,
        )
        
        # 5. Create CCSRUpscaler with extracted components
        self.upscaler = CCSRUpscaler(
            unet=temp_pipe.unet,
            vae=vae,
            controlnet=controlnet,
            scheduler=temp_pipe.scheduler,
            text_encoder=temp_pipe.text_encoder,
            tokenizer=temp_pipe.tokenizer,
            device=self.device,
        )
        self.upscaler.vae.enable_tiling()

        logger.info("Pipeline setup complete.")

    def __call__(self, config: PipelineConfig, workload: BatchTask) -> dict:
        logger.info(
            f"Executing upscale and background removal batch: {workload.task_id}"
        )

        # Determine dynamic batch size limit and chunk workload if needed
        allowed_batch_size = determine_batch_size(config.vram_limit_pct)
        items = workload.items

        # We process the batch
        pil_images = []
        for item in items:
            pil_images.append(Image.open(item.input_path))

        # Step 1: Batched Background Removal (RMBG-2.0)
        logger.info("Step 1: Running batched background removal...")

        image_size = (1024, 1024)
        transform_image = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Batch tensors
        rmbg_inputs = (
            torch.stack([transform_image(img.convert("RGB")) for img in pil_images])
            .to(self.device)
            .to(torch.float32)
        )

        with torch.inference_mode():
            preds = self.rmbg(rmbg_inputs)[-1].sigmoid()  # Get output mask batch

        masks_1024 = []
        for j in range(len(pil_images)):
            pred_mask = preds[j].squeeze().cpu()
            mask_pil = TF.to_pil_image(pred_mask).resize((1024, 1024))
            masks_1024.append(mask_pil)

        # Step 2: CCSR Upscaling (ControlNet + VAE + SD 2.1) -> Target Scale Size
        target_dim = 1024 * config.scale_factor
        target_size = (target_dim, target_dim)
        logger.info(f"Step 2: Running CCSR upscaler (ControlNet SD 2.1) to target size {target_size}...")
        
        # Resize input images to target_size for ControlNet tile conditioning
        control_images = [
            img.convert("RGB").resize(target_size, Image.Resampling.BICUBIC)
            for img in pil_images
        ]
        prompts = [item.prompt for item in items]

        with torch.inference_mode():
            upscaled_images = self.upscaler(
                images=control_images,
                prompts=prompts,
                height=target_dim,
                width=target_dim,
                num_inference_steps=config.ccsr_steps,
                guidance_scale=config.ccsr_guidance_scale,
            )

        # Step 3: GPU Guided Filter mask upscale
        logger.info("Step 3: Refining mask via GPU Guided Filter...")
        refined_masks = self._run_guided_filter(
            upscaled_images, masks_1024, config.guided_radius, config.guided_eps, target_size
        )

        # Step 4: Construct results
        results = {}
        for j, item in enumerate(items):
            # Create merged mask 1024 with original image
            orig_img = pil_images[j]
            original_resized = orig_img.convert("RGB").resize(
                (1024, 1024), Image.Resampling.LANCZOS
            )

            merged_1024 = original_resized.copy()
            merged_1024.putalpha(masks_1024[j])

            results[item.relative_path] = {
                "upscale_4k": upscaled_images[j],
                "mask_1024": masks_1024[j],
                "mask_4096": refined_masks[j],
                "merged_1024": merged_1024,
            }

        return {"dst_root": workload.dst_root, "items": results}

    def _run_guided_filter(
        self,
        pil_guides: List[Image.Image],
        pil_masks_1024: List[Image.Image],
        r: int,
        eps: float,
        target_size: tuple[int, int]
    ) -> List[Image.Image]:
        # Convert PIL images to torch tensors
        guide_tensors = []
        src_tensors = []

        for guide_pil, mask_pil in zip(pil_guides, pil_masks_1024):
            # Resize mask coarsely to target size
            mask_resized = mask_pil.resize(target_size, Image.Resampling.BILINEAR)
            guide_tensors.append(TF.to_tensor(guide_pil.convert("RGB")))
            src_tensors.append(TF.to_tensor(mask_resized))

        guide_batch = torch.stack(guide_tensors).to(self.device)
        src_batch = torch.stack(src_tensors).to(self.device)

        with torch.inference_mode():
            refined_batch = guided_filter_torch(guide_batch, src_batch, r, eps)

        refined_batch = torch.clamp(refined_batch, 0.0, 1.0).cpu()

        refined_pil_images = []
        for j in range(refined_batch.shape[0]):
            refined_pil_images.append(TF.to_pil_image(refined_batch[j]))

        return refined_pil_images
