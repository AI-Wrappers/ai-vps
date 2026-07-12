from pydantic import BaseModel
from typing import List

class PipelineConfig(BaseModel):
    ccsr_steps: int = 4
    ccsr_guidance_scale: float = 1.0
    vram_limit_pct: float = 1.0
    scale_factor: int = 4

class ImageItem(BaseModel):
    input_path: str
    relative_path: str
    prompt: str = ""

class BatchTask(BaseModel):
    task_id: str
    src_root: str
    dst_root: str
    items: List[ImageItem]
