import json
from pathlib import Path
from typing import Dict, Any
from PIL import Image
from ai_pipeline_toolbox.core.interfaces import BaseResultSaver

class ImageGroupResultSaver(BaseResultSaver[Image.Image]):
    """
    Saves PIL.Image to grouped subdirectories based on group_name in metadata.
    """
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: Image.Image, metadata: Dict[str, Any]) -> str:
        task_id = metadata.get("task_id", "unknown_task")
        group_name = task_id.split("___")[0] if "___" in task_id else "unknown"
        task_name = task_id.split("___")[1] if "___" in task_id else task_id
        
        group_dir = self.output_dir / group_name
        group_dir.mkdir(exist_ok=True)
        
        output_path = group_dir / f"{task_name}.png"
        result.save(output_path)
        
        # Save metadata JSON alongside
        meta_path = group_dir / f"{task_name}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        return str(output_path)
