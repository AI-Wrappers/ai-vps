import logging
from pathlib import Path
from typing import Dict, Any
from ai_pipeline_toolbox.core.interfaces import BaseResultSaver

logger = logging.getLogger(__name__)

class MultiArtifactSaver(BaseResultSaver[dict]):
    """
    Saves multiple generated artifacts (upscale_4k, mask_1024, mask_4096, merged_1024)
    maintaining the original relative folder structure under the destination root.
    """
    def save(self, result: dict, metadata: Dict[str, Any]) -> str:
        dst_root = Path(result["dst_root"])
        saved_paths = []

        for rel_path_str, artifacts in result["items"].items():
            rel_path = Path(rel_path_str)
            output_dir = dst_root / rel_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            base_name = rel_path.stem
            # Default to .png if not specified
            ext = rel_path.suffix if rel_path.suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} else ".png"

            # Paths for output files
            upscale_path = output_dir / f"{base_name}_upscale{ext}"
            mask_1024_path = output_dir / f"{base_name}_mask_1024{ext}"
            mask_4096_path = output_dir / f"{base_name}_mask_4096{ext}"
            merged_1024_path = output_dir / f"{base_name}_merged_1024{ext}"

            # Save the Pillow image objects
            artifacts["upscale_4k"].save(upscale_path)
            artifacts["mask_1024"].save(mask_1024_path)
            artifacts["mask_4096"].save(mask_4096_path)
            artifacts["merged_1024"].save(merged_1024_path)

            logger.info(f"Saved artifacts for {rel_path_str} to {output_dir}")
            saved_paths.append(str(upscale_path))

        return saved_paths[0] if saved_paths else str(dst_root)
