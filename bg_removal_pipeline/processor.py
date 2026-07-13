import os
import logging
from pathlib import Path
from typing import Any, Iterable
from ai_pipeline_toolbox.core.interfaces import BaseWorkloadProcessor
from bg_removal_pipeline.schemas import BatchTask, ImageItem

logger = logging.getLogger(__name__)

class SimpleDirectoryWorkloadProcessor(BaseWorkloadProcessor):
    def __init__(self, default_batch_size: int = 4):
        self.default_batch_size = default_batch_size

    def process(self, raw_workload: Any) -> Iterable[BatchTask]:
        if not isinstance(raw_workload, dict):
            raise ValueError("Workload must be a dictionary containing 'src' and 'dst'.")

        src = raw_workload.get("src")
        dst = raw_workload.get("dst")
        if not src or not dst:
            raise ValueError("Workload must specify both 'src' and 'dst' paths.")

        batch_size = raw_workload.get("batch_size", self.default_batch_size)

        src_root = Path(src).resolve()
        dst_root = Path(dst).resolve()

        if not src_root.exists():
            raise FileNotFoundError(f"Source directory does not exist: {src_root}")

        supported_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        all_items = []
        for file_path in sorted(src_root.glob("**/*")):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                if file_path.stem.endswith(("_mask_1024", "_merged_1024")):
                    continue
                rel_path = file_path.relative_to(src_root)
                all_items.append(ImageItem(
                    input_path=str(file_path),
                    relative_path=str(rel_path)
                ))

        if not all_items:
            logger.warning(f"No supported images found in {src_root}")
            return []

        batches = []
        for i in range(0, len(all_items), batch_size):
            chunk = all_items[i:i + batch_size]
            batch_id = f"batch_{i // batch_size:04d}"
            batches.append(BatchTask(
                task_id=batch_id,
                src_root=str(src_root),
                dst_root=str(dst_root),
                items=chunk
            ))

        logger.info(f"Scanned {len(all_items)} images and created {len(batches)} batches (batch size: {batch_size}).")
        return batches
