from pathlib import Path
from typing import Any, Dict
from PIL import Image

from ai_pipeline_toolbox.core.interfaces import BaseResultSaver

class ImageResultSaver(BaseResultSaver):
    """
    Saves a PIL.Image to output_dir respecting the task_id as the relative subpath.
    This reconstructs the directory structure based on task_id (e.g. group_name/image_name).
    """
    def __init__(self, output_dir: str = "outputs", ext: str = ".png"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ext = ext if ext.startswith(".") else f".{ext}"

    def save(self, result: Any, metadata: Dict[str, Any]) -> str:
        """
        result: A PIL.Image object.
        metadata: Dictionary containing 'task_id'.
        """
        if not isinstance(result, Image.Image):
            raise TypeError("Result must be a PIL.Image object.")
            
        task_id = metadata.get("task_id", "unknown_task")
        
        # Resolve the relative path based on task_id
        output_path = self.output_dir / f"{task_id}{self.ext}"
        
        # Ensure parent directories exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the image
        result.save(output_path)
        
        return str(output_path)
