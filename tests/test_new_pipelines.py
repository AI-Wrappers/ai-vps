import pytest
import os
import tempfile
import json
import torch
from PIL import Image
from pathlib import Path

from bg_removal_pipeline.processor import SimpleDirectoryWorkloadProcessor
from ccsr_upscale_pipeline.processor import DirectoryPromptWorkloadProcessor
from guided_filter_pipeline.processor import MaskPairWorkloadProcessor

from guided_filter_pipeline.pipeline import GuidedFilterPipeline
from guided_filter_pipeline.schemas import PipelineConfig, BatchTask, MaskPairItem


@pytest.fixture
def mock_directories(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    
    group_dir = src_dir / "group1"
    group_dir.mkdir()
    
    for i in range(3):
        img_path = group_dir / f"img_{i}.png"
        Image.new("RGB", (100, 100), color="red").save(img_path)
        
    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()
    
    return src_dir, dst_dir

def test_bg_removal_processor(mock_directories):
    src, dst = mock_directories
    processor = SimpleDirectoryWorkloadProcessor(default_batch_size=2)
    workload = {"src": str(src), "dst": str(dst)}
    
    batches = list(processor.process(workload))
    
    assert len(batches) == 2
    assert len(batches[0].items) == 2
    assert len(batches[1].items) == 1
    
    assert batches[0].items[0].relative_path.startswith("group1")

def test_ccsr_upscale_processor(mock_directories):
    src, dst = mock_directories
    processor = DirectoryPromptWorkloadProcessor(default_batch_size=2)
    
    flux_workload = {
        "groups": [
            {
                "group_name": "group1",
                "prompts": [
                    {"name": "img_0", "pos": "A red square"}
                ]
            }
        ]
    }
    
    workload = {
        "src": str(src),
        "dst": str(dst),
        "flux_workload": flux_workload
    }
    
    batches = list(processor.process(workload))
    
    assert len(batches) == 2
    
    img_0_item = next(item for item in batches[0].items if item.relative_path.endswith("img_0.png"))
    assert img_0_item.prompt == "A red square"
    
    img_1_item = next(item for item in batches[0].items if item.relative_path.endswith("img_1.png"))
    assert img_1_item.prompt == ""

def test_guided_filter_pipeline_and_processor(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    
    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()
    
    mask_path = masks_dir / "test_mask_1024.png"
    Image.new("L", (128, 128), color=255).save(mask_path)
    
    img_path = images_dir / "test_upscale_4k.png"
    Image.new("RGB", (512, 512), color="blue").save(img_path)
    
    processor = MaskPairWorkloadProcessor(default_batch_size=2)
    workload = {
        "src_masks": str(masks_dir),
        "src_images": str(images_dir),
        "dst": str(dst_dir)
    }
    
    batches = list(processor.process(workload))
    
    assert len(batches) == 1
    assert len(batches[0].items) == 1
    
    item = batches[0].items[0]
    assert item.mask_1k_path == str(mask_path)
    assert item.upscale_4k_path == str(img_path)
    
    pipeline = GuidedFilterPipeline()
    pipeline.setup({})
    
    config = PipelineConfig(vram_limit_pct=1.0)
    
    result = pipeline(config, batches[0])
    
    assert "dst_root" in result
    assert "items" in result
    
    output_item = result["items"]["test"]
    assert "mask_4096" in output_item
    assert "merged_4096" in output_item
    
    out_mask = output_item["mask_4096"]
    assert out_mask.size == (512, 512)
