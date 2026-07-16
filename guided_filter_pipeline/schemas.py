from pydantic import BaseModel
from typing import List

class PipelineConfig(BaseModel):
    guided_radius: int = 4
    guided_eps: float = 1e-6
    vram_limit_pct: float = 1.0

class MaskPairItem(BaseModel):
    mask_1k_path: str
    upscale_4k_path: str
    relative_path: str

class BatchTask(BaseModel):
    task_id: str
    src_masks_root: str
    src_images_root: str
    dst_root: str
    items: List[MaskPairItem]
