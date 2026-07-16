import os
import torch
import logging
from typing import Dict, List, Union
from enum import Enum
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from transformers import AutoConfig, AutoModelForImageSegmentation
from safetensors.torch import load_file

from ai_pipeline_toolbox.core.pipeline import BaseGenerationPipeline
from ai_pipeline_toolbox.registry.generated_enums import RmbgModels

from bg_removal_pipeline.schemas import PipelineConfig, BatchTask

logger = logging.getLogger(__name__)

def determine_batch_size(vram_limit_pct: float) -> int:
    if not torch.cuda.is_available():
        return 1

    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_vram_gb = total_memory / (1024**3)
    allowed_vram_gb = total_vram_gb * vram_limit_pct

    weights_footprint_gb = 2.0
    activation_per_item_gb = 0.5

    available_vram = allowed_vram_gb - weights_footprint_gb
    if available_vram <= 0:
        return 1

    batch_size = max(1, int(available_vram / activation_per_item_gb))
    logger.info(f"Dynamic VRAM Batch Size Calculation (RMBG): Optimal Batch Size: {batch_size}")
    return batch_size

class BgRemovalPipeline(BaseGenerationPipeline[PipelineConfig, BatchTask, dict]):
    required_models = [RmbgModels.BRIIA_RMBG_V2]

    def setup(self, models_paths: Dict[Union[Enum, str], str]) -> None:
        logger.info("Setting up BgRemovalPipeline components...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        rmbg_path = models_paths[RmbgModels.BRIIA_RMBG_V2]
        logger.info(f"Loading RMBG-2.0 weights from {rmbg_path}...")
        config = AutoConfig.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)
        self.rmbg = AutoModelForImageSegmentation.from_config(
            config, trust_remote_code=True
        )
        self.rmbg.load_state_dict(load_file(rmbg_path))
        self.rmbg.to(self.device).eval()

        logger.info("Pipeline setup complete.")

    def __call__(self, config: PipelineConfig, workload: BatchTask) -> dict:
        logger.info(f"Executing background removal batch: {workload.task_id}")
        
        # Override batch size dynamically if needed based on the logic:
        # We can dynamically enforce a smaller batch size here based on vram limits, 
        # but since batching was decided at the processor level, we can either re-batch or just warn.
        allowed_batch_size = determine_batch_size(config.vram_limit_pct)
        items = workload.items
        
        if len(items) > allowed_batch_size:
             logger.warning(f"Batch size {len(items)} exceeds allowed VRAM safe size {allowed_batch_size}.")

        pil_images = [Image.open(item.input_path) for item in items]

        image_size = (1024, 1024)
        transform_image = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        rmbg_inputs = (
            torch.stack([transform_image(img.convert("RGB")) for img in pil_images])
            .to(self.device)
            .to(torch.float32)
        )

        with torch.inference_mode():
            preds = self.rmbg(rmbg_inputs)[-1].sigmoid()

        masks_1024 = []
        for j in range(len(pil_images)):
            pred_mask = preds[j].squeeze().cpu()
            mask_pil = TF.to_pil_image(pred_mask).resize((1024, 1024))
            masks_1024.append(mask_pil)

        results = {}
        for j, item in enumerate(items):
            orig_img = pil_images[j]
            original_resized = orig_img.convert("RGB").resize(
                (1024, 1024), Image.Resampling.LANCZOS
            )
            merged_1024 = original_resized.copy()
            merged_1024.putalpha(masks_1024[j])

            results[item.relative_path] = {
                "mask_1024": masks_1024[j],
                "merged_1024": merged_1024,
            }

        return {"dst_root": workload.dst_root, "items": results}
