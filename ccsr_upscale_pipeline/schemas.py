from pydantic import BaseModel
from typing import List


class PipelineConfig(BaseModel):
    seed: int
    scale_factor: int = 4


class ImageItem(BaseModel):
    input_path: str
    relative_path: str
    prompt: str = ""
    negative_prompt: str = ""
    parent_id: str = ""
    file_id: str = ""


class SingleTask(BaseModel):
    task_id: str
    src_root: str
    dst_root: str
    item: ImageItem
