import os
import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Union
from enum import Enum
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from ai_pipeline_toolbox.core.pipeline import BaseGenerationPipeline
from guided_filter_pipeline.schemas import PipelineConfig, BatchTask

logger = logging.getLogger(__name__)

def guided_filter_torch(guide: torch.Tensor, src: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    box_size = 2 * r + 1
    
    if guide.shape[1] == 3:
        guide_gray = (
            0.299 * guide[:, 0:1] + 0.587 * guide[:, 1:2] + 0.114 * guide[:, 2:3]
        )
    else:
        guide_gray = guide

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
    if not torch.cuda.is_available():
        return 16

    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_vram_gb = total_memory / (1024**3)
    allowed_vram_gb = total_vram_gb * vram_limit_pct

    activation_per_item_gb = 0.15
    batch_size = max(1, int(allowed_vram_gb / activation_per_item_gb))
    
    batch_size = min(batch_size, 32)
    logger.info(f"Dynamic VRAM Batch Size Calculation (Guided Filter): Optimal Batch Size: {batch_size}")
    return batch_size


class GuidedFilterPipeline(BaseGenerationPipeline[PipelineConfig, BatchTask, dict]):
    required_models = []

    def setup(self, models_paths: Dict[Union[Enum, str], str]) -> None:
        logger.info("Setting up GuidedFilterPipeline components...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Pipeline setup complete.")

    def __call__(self, config: PipelineConfig, workload: BatchTask) -> dict:
        logger.info(f"Executing Guided Filter batch: {workload.task_id}")

        allowed_batch_size = determine_batch_size(config.vram_limit_pct)
        items = workload.items
        
        if len(items) > allowed_batch_size:
            logger.warning(f"Batch size {len(items)} exceeds allowed VRAM safe size {allowed_batch_size}.")

        pil_guides = [Image.open(item.upscale_4k_path) for item in items]
        pil_masks_1k = [Image.open(item.mask_1k_path).convert("L") for item in items]
        
        target_sizes = [g.size for g in pil_guides] 

        refined_masks = self._run_guided_filter(
            pil_guides, pil_masks_1k, config.guided_radius, config.guided_eps, target_sizes[0]
        )

        results = {}
        for j, item in enumerate(items):
            merged_4096 = pil_guides[j].copy().convert("RGBA")
            merged_4096.putalpha(refined_masks[j])

            results[item.relative_path] = {
                "mask_4096": refined_masks[j],
                "merged_4096": merged_4096,
            }

        return {"dst_root": workload.dst_root, "items": results}

    def _run_guided_filter(
        self,
        pil_guides: List[Image.Image],
        pil_masks_1k: List[Image.Image],
        r: int,
        eps: float,
        target_size: tuple[int, int]
    ) -> List[Image.Image]:
        guide_tensors = []
        src_tensors = []

        for guide_pil, mask_pil in zip(pil_guides, pil_masks_1k):
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
