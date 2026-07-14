import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Iterable
from ai_pipeline_toolbox.core.interfaces import BaseWorkloadProcessor
from ccsr_upscale_pipeline.schemas import SingleTask, ImageItem
from ccsr_upscale_pipeline.gdrive_utils import GDriveClient

logger = logging.getLogger(__name__)

class DirectoryPromptWorkloadProcessor(BaseWorkloadProcessor):
    def __init__(self, default_batch_size: int = 4):
        self.default_batch_size = default_batch_size
        self.gdrive = GDriveClient()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ccsr_inputs_"))

    def process(self, raw_workload: Any) -> Iterable[SingleTask]:
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

        prompt_lookup = {}
        # Parse the JSON structure properly based on user's confirmation
        groups = flux_workload if isinstance(flux_workload, list) else flux_workload.get("groups", [])
        for group in groups:
            g_name = group.get("group_name")
            if not g_name:
                continue
            for p in group.get("prompts", []):
                p_name = p.get("name")
                if not p_name:
                    continue
                prompt_lookup[(g_name, p_name)] = {
                    "pos": p.get("pos", ""),
                    "neg": p.get("neg", "")
                }

        logger.info(f"Scanning Google Drive source folder: {src}")
        gdrive_files = self.gdrive.list_files_recursively(src)

        supported_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        matched_files = []
        
        for file_info in gdrive_files:
            rel_path = file_info['rel_path']
            suffix = rel_path.suffix.lower()
            if suffix not in supported_extensions:
                continue
                
            if len(rel_path.parts) < 2:
                continue # Needs at least group_name/file_name
                
            g_name = rel_path.parent.name
            img_name = rel_path.stem
            
            if (g_name, img_name) in prompt_lookup:
                matched_prompt = prompt_lookup[(g_name, img_name)]["pos"]
                matched_neg_prompt = prompt_lookup[(g_name, img_name)]["neg"]
                matched_files.append((file_info, matched_prompt, matched_neg_prompt))
            else:
                logger.debug(f"Skipping {rel_path} - no matching prompt found.")

        if not matched_files:
            logger.warning(f"No supported images with matching prompts found in {src}")
            return []

        logger.info(f"Found {len(matched_files)} matching images. Processing one by one.")

        for i, (file_info, matched_prompt, matched_neg_prompt) in enumerate(matched_files):
            task_id = f"task_{i:04d}"
            
            file_id = file_info['id']
            rel_path = file_info['rel_path']
            parent_id = file_info['parent_id']
            
            local_dir = self.temp_dir / rel_path.parent
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / rel_path.name
            
            self.gdrive.download_file(file_id, local_path)
            
            item = ImageItem(
                input_path=str(local_path),
                relative_path=str(rel_path),
                prompt=matched_prompt,
                negative_prompt=matched_neg_prompt,
                parent_id=parent_id
            )
            
            yield SingleTask(
                task_id=task_id,
                src_root=src,
                dst_root=dst,
                item=item
            )
