import os
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from pathlib import Path

from ai_pipeline_toolbox.orchestrator.runner import Runner
from ai_pipeline_toolbox.components.state_manager import SQLiteStateManager
from ai_pipeline_toolbox.components.model_fetcher import ModelFetcher
from ai_pipeline_toolbox.components.loop_manager import LoopManager

from upscale_rmbg_pipeline.schemas import PipelineConfig
from upscale_rmbg_pipeline.processor import DirectoryWorkloadProcessor
from upscale_rmbg_pipeline.saver import MultiArtifactSaver
from upscale_rmbg_pipeline.pipeline import BgRemovalUpscalePipeline

@patch('upscale_rmbg_pipeline.pipeline.StableDiffusionControlNetPipeline')
@patch('upscale_rmbg_pipeline.pipeline.ControlNetModel')
@patch('upscale_rmbg_pipeline.pipeline.AutoencoderKL')
@patch('upscale_rmbg_pipeline.pipeline.AutoModelForImageSegmentation')
@patch('upscale_rmbg_pipeline.pipeline.load_file')
@patch('upscale_rmbg_pipeline.pipeline.AutoConfig')
def test_upscale_rmbg_pipeline_flow(
    mock_auto_config, mock_load_file, mock_rmbg_cls, mock_vae_cls, mock_cn_cls, mock_sd_pipeline, tmp_path
):
    # 1. Create dummy input directories and images
    src_dir = tmp_path / "input_images"
    src_dir.mkdir()
    group1 = src_dir / "group1"
    group1.mkdir()
    
    # Create two dummy images
    dummy_img1 = Image.new('RGB', (100, 100), color='red')
    dummy_img1.save(group1 / "img1.png")
    
    dummy_img2 = Image.new('RGB', (200, 200), color='green')
    dummy_img2.save(group1 / "img2.png")
    
    dst_dir = tmp_path / "output_images"
    
    # 2. Mock model weights fetching
    def mock_fetch_func(models):
        return {m: "/dummy/path/model.safetensors" for m in models}
        
    mock_fetcher = MagicMock(spec=ModelFetcher)
    mock_fetcher.fetch.side_effect = mock_fetch_func
    
    # 3. Mock the deep learning models behavior
    # Mock RMBG-2.0
    mock_rmbg_instance = MagicMock()
    import torch
    dummy_preds = torch.zeros([2, 1, 1024, 1024])
    mock_rmbg_instance.return_value = [dummy_preds]
    mock_rmbg_cls.from_pretrained.return_value = mock_rmbg_instance
    
    # Mock VAE and ControlNet loading
    mock_cn_cls.from_single_file.return_value = MagicMock()
    mock_vae_cls.from_single_file.return_value = MagicMock()
    
    # Mock SD ControlNet Pipeline
    mock_pipe_instance = MagicMock()
    mock_output = MagicMock()
    mock_output.images = [
        Image.new('RGB', (4096, 4096), color='blue'),
        Image.new('RGB', (4096, 4096), color='yellow')
    ]
    mock_pipe_instance.return_value = mock_output
    mock_sd_pipeline.from_single_file.return_value = mock_pipe_instance
    
    # 4. Setup runner
    db_path = tmp_path / "state.db"
    
    runner = Runner(
        workload_processor=DirectoryWorkloadProcessor(default_batch_size=2),
        state_manager=SQLiteStateManager(str(db_path)),
        fetcher=mock_fetcher,
        loop_manager=LoopManager(),
        result_saver=MultiArtifactSaver()
    )
    
    config = PipelineConfig(vram_limit_pct=0.5, ccsr_steps=2, scale_factor=4)
    pipeline = BgRemovalUpscalePipeline()
    
    # Merged workload mock
    raw_workload = {
        "flux_workload": {
            "groups": [
                {
                    "group_name": "group1",
                    "prompts": [
                        {"name": "img1", "pos": "A red apple prompt"},
                        {"name": "img2", "pos": "A green leaf prompt"}
                    ]
                }
            ]
        },
        "src": str(src_dir),
        "dst": str(dst_dir),
        "scale_factor": 4,
        "batch_size": 2
    }
    
    # 5. Run execution
    runner.run(pipeline=pipeline, raw_workload=raw_workload, config=config)
    
    # 6. Verify that files are correctly generated in output directory
    expected_files = [
        "group1/img1_upscale.png",
        "group1/img1_mask_1024.png",
        "group1/img1_mask_4096.png",
        "group1/img1_merged_1024.png",
        "group1/img2_upscale.png",
        "group1/img2_mask_1024.png",
        "group1/img2_mask_4096.png",
        "group1/img2_merged_1024.png"
    ]
    
    for rel_f in expected_files:
        p = dst_dir / rel_f
        assert p.exists(), f"Output file {rel_f} was not created!"
        
    # Check that state manager has completed the batch
    state_mgr = SQLiteStateManager(str(db_path))
    assert state_mgr.is_completed("batch_0000") == True
    
    # Verify that the prompts and dynamic target sizes were passed correctly
    args, kwargs = mock_pipe_instance.call_args
    assert kwargs["prompt"] == ["A red apple prompt", "A green leaf prompt"]
    assert kwargs["height"] == 4096
    assert kwargs["width"] == 4096
