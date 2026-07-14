from pydantic import BaseModel
from typing import List

class PipelineConfig(BaseModel):
    ccsr_steps: int = 45
    ccsr_guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    eta: float = 0.0
    color_fix_type: str = "adain"
    tile_stride_ratio: float = 0.5
    seed: int = 42
    vram_limit_pct: float = 1.0
    scale_factor: int = 4

class ImageItem(BaseModel):
    input_path: str
    relative_path: str
    prompt: str = ""
    negative_prompt: str = ""
    parent_id: str = ""

class SingleTask(BaseModel):
    task_id: str
    src_root: str
    dst_root: str
    item: ImageItem
