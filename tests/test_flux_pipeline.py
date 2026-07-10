import os
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from pathlib import Path

from ai_pipeline_toolbox.orchestrator.runner import Runner
from ai_pipeline_toolbox.components.state_manager import SQLiteStateManager
from ai_pipeline_toolbox.components.model_fetcher import ModelFetcher
from ai_pipeline_toolbox.components.loop_manager import LoopManager

from flux_hf_pipeline.schemas import FluxConfig
from flux_hf_pipeline.processor import GroupedWorkloadProcessor
from flux_hf_pipeline.saver import ImageGroupResultSaver
from flux_hf_pipeline.pipeline import Flux1DPipeline

@patch('flux_hf_pipeline.pipeline.FluxPipeline')
@patch('diffusers.FluxTransformer2DModel')
@patch('diffusers.AutoencoderKL')
@patch('transformers.CLIPTextModel')
@patch('transformers.T5EncoderModel')
@patch('safetensors.torch.load_file')
def test_flux_pipeline_flow(mock_load_file, mock_t5, mock_clip, mock_vae, mock_transformer, mock_flux_pipeline, tmp_path):
    # 1. Mock the heavy dependencies
    def mock_fetch_func(models):
        return {m: "/dummy/path/model.safetensors" for m in models}
    
    mock_fetcher = MagicMock(spec=ModelFetcher)
    mock_fetcher.fetch.side_effect = mock_fetch_func
    
    # Mock load_file to return dummy dict
    mock_load_file.return_value = {}
    
    # Mock the pipeline HF model
    mock_pipe_instance = MagicMock()
    dummy_image = Image.new('RGB', (64, 64), color='blue')
    mock_output = MagicMock()
    mock_output.images = [dummy_image]
    mock_pipe_instance.return_value = mock_output
    mock_flux_pipeline.from_pretrained.return_value = mock_pipe_instance

    # 2. Setup the test workload
    raw_workload = {
      "groups": [
        {
          "group_name": "test_group",
          "lora": {
            "name": "test_lora",
            "urn": "urn:air:flux1:lora:civitai:1234@5678",
            "url": "https://example.com/lora.safetensors"
          },
          "prompts": [
            {
              "name": "test_prompt_1",
              "pos": "A test prompt"
            }
          ]
        }
      ]
    }

    # 3. Setup temporary directories for outputs and state
    db_path = tmp_path / "state.db"
    outputs_dir = tmp_path / "outputs"

    runner = Runner(
        workload_processor=GroupedWorkloadProcessor(),
        state_manager=SQLiteStateManager(str(db_path)),
        fetcher=mock_fetcher,
        loop_manager=LoopManager(),
        result_saver=ImageGroupResultSaver(str(outputs_dir))
    )
    
    config = FluxConfig(num_inference_steps=2, guidance_scale=3.5)
    pipeline = Flux1DPipeline()
    
    # 4. Execute the orchestrator
    with patch('torchao.quantization.quantize_') as mock_quantize:
        runner.run(pipeline=pipeline, raw_workload=raw_workload, config=config)
    
    # 5. Assertions
    mock_flux_pipeline.from_pretrained.assert_called_once()
    mock_pipe_instance.assert_called_once()
    
    expected_image = outputs_dir / "test_group" / "test_prompt_1.png"
    expected_meta = outputs_dir / "test_group" / "test_prompt_1.json"
    
    assert expected_image.exists(), "Image was not saved in the correct group directory"
    assert expected_meta.exists(), "Metadata JSON was not saved"
    
    state_mgr = SQLiteStateManager(str(db_path))
    assert state_mgr.is_completed("test_group___test_prompt_1") == True
