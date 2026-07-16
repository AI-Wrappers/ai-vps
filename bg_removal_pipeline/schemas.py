from pydantic import BaseModel
from typing import List

class PipelineConfig(BaseModel):
    vram_limit_pct: float = 1.0

class ImageItem(BaseModel):
    input_path: str
    relative_path: str

class BatchTask(BaseModel):
    task_id: str
    src_root: str
    dst_root: str
    items: List[ImageItem]
