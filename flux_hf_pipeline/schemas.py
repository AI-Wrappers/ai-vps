from typing import Optional, List
from pydantic import BaseModel, Field

class LoraInfo(BaseModel):
    name: str
    urn: str
    url: str
    trigger_words: List[str] = Field(default_factory=list)

class FluxTask(BaseModel):
    task_id: str
    group_name: str
    name: str
    prompt: str
    lora: Optional[LoraInfo] = None

class FluxConfig(BaseModel):
    num_inference_steps: int = Field(default=50, ge=1, le=150)
    guidance_scale: float = Field(default=3.5, ge=1.0, le=20.0)
    height: int = Field(default=1024)
    width: int = Field(default=1024)
    max_sequence_length: int = Field(default=512)
    seed: int = Field(default=0)
