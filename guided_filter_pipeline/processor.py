import os
import logging
from pathlib import Path
from typing import Any, Iterable
from ai_pipeline_toolbox.core.interfaces import BaseWorkloadProcessor
from guided_filter_pipeline.schemas import BatchTask, MaskPairItem

logger = logging.getLogger(__name__)


class MaskPairWorkloadProcessor(BaseWorkloadProcessor):
    def __init__(self, default_batch_size: int = 4):
        self.default_batch_size = default_batch_size

    def process(self, raw_workload: Any) -> Iterable[BatchTask]:
        if not isinstance(raw_workload, dict):
            raise ValueError(
                "Workload must be a dictionary containing 'src_masks', 'src_images', and 'dst'."
            )

        src_masks = raw_workload.get("src_masks")
        src_images = raw_workload.get("src_images")
        dst = raw_workload.get("dst")
        if not src_masks or not src_images or not dst:
            raise ValueError(
                "Workload must specify 'src_masks', 'src_images' and 'dst' paths."
            )

        batch_size = raw_workload.get("batch_size", self.default_batch_size)

        masks_root = Path(src_masks).resolve()
        images_root = Path(src_images).resolve()
        dst_root = Path(dst).resolve()

        if not masks_root.exists() or not images_root.exists():
            raise FileNotFoundError(f"Source directories do not exist.")

        supported_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        all_items = []

        for mask_path in sorted(masks_root.glob("**/*")):
            if mask_path.is_file() and mask_path.suffix.lower() in supported_extensions:
                if "_mask_1024" in mask_path.name:
                    base_name = mask_path.name.replace(
                        "_mask_1024" + mask_path.suffix, ""
                    )
                    rel_dir = mask_path.relative_to(masks_root).parent

                    img_path = None
                    for suffix in ["_upscale_4k", "_upscaled"]:
                        for ext in [mask_path.suffix, ".png", ".jpg", ".jpeg", ".webp"]:
                            candidate = (
                                images_root / rel_dir / f"{base_name}{suffix}{ext}"
                            )
                            if candidate.exists():
                                img_path = candidate
                                break
                        if img_path:
                            break

                    if img_path:
                        all_items.append(
                            MaskPairItem(
                                mask_1k_path=str(mask_path),
                                upscale_4k_path=str(img_path),
                                relative_path=str(rel_dir / base_name),
                            )
                        )
                    else:
                        logger.warning(
                            f"Found mask {mask_path} but missing upscaled image under {images_root / rel_dir}"
                        )

        if not all_items:
            logger.warning(f"No paired masks and images found.")
            return []

        batches = []
        for i in range(0, len(all_items), batch_size):
            chunk = all_items[i : i + batch_size]
            batch_id = f"batch_{i // batch_size:04d}"
            batches.append(
                BatchTask(
                    task_id=batch_id,
                    src_masks_root=str(masks_root),
                    src_images_root=str(images_root),
                    dst_root=str(dst_root),
                    items=chunk,
                )
            )

        logger.info(
            f"Scanned and created {len(batches)} batches (batch size: {batch_size})."
        )
        return batches
