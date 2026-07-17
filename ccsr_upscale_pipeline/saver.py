import os
import io
import logging
from pathlib import Path
from ai_pipeline_toolbox.core.interfaces import BaseResultSaver

logger = logging.getLogger(__name__)

class CcsrUpscaleResultSaver(BaseResultSaver):
    def __init__(self):
        pass

    def save(self, result: dict, meta) -> None:
        import json
        from ccsr_upscale_pipeline.gdrive_utils import GDriveTransferManager
        
        items = result.get("items", {})
        dst_root = result.get("dst_root", "data/outputs_ccsr")

        for rel_path_str, data in items.items():
            rel_path = Path(rel_path_str)
            base_name = rel_path.stem
            
            item = data.get("item")
            upscale_img = data.get("upscale_4k")
            
            if not item or not upscale_img:
                logger.warning(f"Missing item or upscale_4k data for {rel_path_str}")
                continue

            out_name = f"{base_name}_upscaled.png"
            local_dir = Path(dst_root) / Path(item.relative_path).parent
            local_dir.mkdir(parents=True, exist_ok=True)
            
            local_png_path = local_dir / out_name
            local_json_path = local_dir / f"{base_name}_upscaled.json"

            # Save PNG locally
            try:
                upscale_img.save(local_png_path, format='PNG')
                logger.info(f"Successfully saved {out_name} locally to {local_png_path}")
            except Exception as e:
                logger.error(f"Failed to save upscaled PNG locally to {local_png_path}: {e}", exc_info=True)
                continue

            # Write JSON metadata
            try:
                metadata = {
                    "parent_id": item.parent_id,
                    "relative_path": item.relative_path
                }
                with open(local_json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4, ensure_ascii=False)
                logger.debug(f"Saved upload metadata to {local_json_path}")
            except Exception as e:
                logger.error(f"Failed to write metadata JSON to {local_json_path}: {e}", exc_info=True)
                # If metadata fails, delete png so we don't have dangling file
                local_png_path.unlink(missing_ok=True)
                continue

            # Clean up the temporary downloaded input file
            input_path = Path(item.input_path)
            if input_path.exists():
                try:
                    os.remove(input_path)
                    logger.debug(f"Cleaned up temporary input file: {input_path}")
                except Exception as e:
                    logger.warning(f"Could not delete temporary input file {input_path}: {e}")

            # Notify the transfer manager to start the upload backlog task immediately
            transfer_manager = GDriveTransferManager(dst_root=dst_root)
            transfer_manager.trigger_scan()
