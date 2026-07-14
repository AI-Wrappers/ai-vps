import os
import io
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ai_pipeline_toolbox.core.interfaces import BaseResultSaver
from ccsr_upscale_pipeline.gdrive_utils import GDriveClient

logger = logging.getLogger(__name__)

class CcsrUpscaleResultSaver(BaseResultSaver):
    def __init__(self, max_workers: int = 4):
        self.gdrive = GDriveClient()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _upload_and_cleanup(self, item, img_data, base_name: str) -> None:
        try:
            out_name = f"{base_name}_upscale_4k.png"
            parent_id = item.parent_id
            
            if not parent_id:
                logger.error(f"Cannot upload {out_name}, missing parent_id in item")
                return
                
            img_byte_arr = io.BytesIO()
            img_data.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            file_id = self.gdrive.upload_file(parent_id, out_name, img_byte_arr)
            logger.info(f"Successfully uploaded {out_name} to Google Drive folder {parent_id} with ID: {file_id}")
            
        except Exception as e:
            logger.error(f"Failed to upload {base_name}: {e}", exc_info=True)
        finally:
            input_path = Path(item.input_path)
            if input_path.exists():
                try:
                    os.remove(input_path)
                    logger.debug(f"Cleaned up temporary input file: {input_path}")
                except Exception as e:
                    logger.warning(f"Could not delete temporary input file {input_path}: {e}")

    def save(self, result: dict, meta) -> None:
        items = result.get("items", {})

        for rel_path_str, data in items.items():
            rel_path = Path(rel_path_str)
            base_name = rel_path.stem
            
            item = data.get("item")
            upscale_img = data.get("upscale_4k")
            
            if not item or not upscale_img:
                logger.warning(f"Missing item or upscale_4k data for {rel_path_str}")
                continue

            self.executor.submit(self._upload_and_cleanup, item, upscale_img, base_name)
