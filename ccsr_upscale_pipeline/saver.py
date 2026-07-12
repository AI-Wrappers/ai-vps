import os
import logging
from pathlib import Path
from ai_pipeline_toolbox.core.interfaces import BaseResultSaver

logger = logging.getLogger(__name__)

class CcsrUpscaleResultSaver(BaseResultSaver):
    def save(self, result: dict) -> None:
        dst_root = Path(result["dst_root"])
        items = result["items"]

        for rel_path_str, data in items.items():
            rel_path = Path(rel_path_str)
            base_name = rel_path.stem
            parent_dir = rel_path.parent
            
            out_dir = dst_root / parent_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            
            upscale_path = out_dir / f"{base_name}_upscale_4k.png"
            data["upscale_4k"].save(upscale_path, format="PNG")
            
            logger.info(f"Saved {base_name} outputs to {out_dir}")
