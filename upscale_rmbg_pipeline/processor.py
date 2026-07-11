import os
import json
import logging
from pathlib import Path
from typing import Any, Iterable
from ai_pipeline_toolbox.core.interfaces import BaseWorkloadProcessor
from upscale_rmbg_pipeline.schemas import BatchTask, ImageItem

logger = logging.getLogger(__name__)

class DirectoryWorkloadProcessor(BaseWorkloadProcessor):
    """
    Scans the source directory and chunks images into BatchTask items,
    matching each image to its respective prompt defined in the Flux workload JSON.
    """
    def __init__(self, default_batch_size: int = 4):
        self.default_batch_size = default_batch_size

    def process(self, raw_workload: Any) -> Iterable[BatchTask]:
        if isinstance(raw_workload, str):
            try:
                raw_workload = json.loads(raw_workload)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}")

        if not isinstance(raw_workload, dict):
            raise ValueError("Workload must be a dictionary containing 'src' and 'dst'.")

        src = raw_workload.get("src")
        dst = raw_workload.get("dst")
        if not src or not dst:
            raise ValueError("Workload must specify both 'src' and 'dst' paths.")

        batch_size = raw_workload.get("batch_size", self.default_batch_size)
        flux_workload = raw_workload.get("flux_workload", {})

        # Build prompt lookup: (group_name, prompt_name) -> prompt text
        prompt_lookup = {}
        groups = flux_workload.get("groups", [])
        for group in groups:
            g_name = group.get("group_name", "default_group")
            for p in group.get("prompts", []):
                p_name = p.get("name", "")
                p_prompt = p.get("pos", "")
                prompt_lookup[(g_name, p_name)] = p_prompt
                # Also fall back to mapping by name alone in case structure is flat
                prompt_lookup[p_name] = p_prompt

        src_root = Path(src).resolve()
        dst_root = Path(dst).resolve()

        if not src_root.exists():
            raise FileNotFoundError(f"Source directory does not exist: {src_root}")

        supported_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        all_items = []
        # Use recursion to find all files and sort to guarantee deterministic order
        for file_path in sorted(src_root.glob("**/*")):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                rel_path = file_path.relative_to(src_root)
                
                # Match prompt based on parent folder name (group) and filename (prompt name)
                g_name = rel_path.parent.name
                img_name = rel_path.stem
                
                # Search using structured key, falling back to stem name, then empty string
                matched_prompt = prompt_lookup.get((g_name, img_name)) or prompt_lookup.get(img_name) or ""
                
                all_items.append(ImageItem(
                    input_path=str(file_path),
                    relative_path=str(rel_path),
                    prompt=matched_prompt
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
