import sys
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
from flux_comfy_lib_pipeline.pipeline import FluxComfyLibPipeline

# Mock all ComfyUI modules during imports to avoid requiring ComfyUI installation to run tests
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.sd'] = MagicMock()
sys.modules['comfy.cli_args'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.sample'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['comfy.options'] = MagicMock()
sys.modules['comfy_extras'] = MagicMock()
sys.modules['comfy_extras.nodes_model_advanced'] = MagicMock()
sys.modules['nodes'] = MagicMock()

@patch('comfy.sd.load_diffusion_model')
@patch('comfy.sd.load_clip')
@patch('comfy.sd.VAE')
@patch('comfy.sd.load_lora_for_models')
@patch('comfy_extras.nodes_model_advanced.ModelSamplingFlux')
@patch('nodes.common_ksampler')
def test_comfy_lib_pipeline_flow(
    mock_ksampler, mock_ms_flux, mock_load_lora, mock_vae_cls, mock_load_clip, mock_load_dit, tmp_path
):
    # 1. Mock fetcher
    def mock_fetch_func(models):
        return {m: "/dummy/path/model.safetensors" for m in models}
    
    mock_fetcher = MagicMock(spec=ModelFetcher)
    mock_fetcher.fetch.side_effect = mock_fetch_func

    # Mock ComfyUI return values
    mock_model = MagicMock()
    mock_clip_model = MagicMock()
    mock_vae_model = MagicMock()
    
    # Setup return values on both sys.modules mock instances to cover all import resolution paths
    mock_load_dit.return_value = mock_model
    sys.modules['comfy'].sd.load_diffusion_model.return_value = mock_model
    
    mock_load_clip.return_value = mock_clip_model
    sys.modules['comfy'].sd.load_clip.return_value = mock_clip_model
    
    mock_vae_cls.return_value = mock_vae_model
    sys.modules['comfy'].sd.VAE.return_value = mock_vae_model
    
    # Mock LoRA application
    mock_load_lora.return_value = (mock_model, mock_clip_model)
    sys.modules['comfy'].sd.load_lora_for_models.return_value = (mock_model, mock_clip_model)
    
    # Mock ModelSamplingFlux
    mock_ms_instance = MagicMock()
    mock_ms_instance.patch.return_value = (mock_model,)
    mock_ms_flux.return_value = mock_ms_instance
    sys.modules['comfy_extras'].nodes_model_advanced.ModelSamplingFlux.return_value = mock_ms_instance
    
    # Mock CLIP tokenize and encoding
    mock_clip_model.tokenize.return_value = {}
    mock_clip_model.encode_from_tokens_scheduled.return_value = [MagicMock()]
    
    # Mock common_ksampler output
    mock_sampled_latent = {"samples": MagicMock()}
    mock_ksampler.return_value = (mock_sampled_latent,)
    sys.modules['nodes'].common_ksampler.return_value = (mock_sampled_latent,)
    
    # Mock VAE decode output tensor
    import torch
    dummy_tensor = torch.zeros([1, 64, 64, 3])
    mock_vae_model.decode.return_value = dummy_tensor
    
    # Configure model_management mocks for device and dtype
    sys.modules['comfy'].model_management.intermediate_device.return_value = "cpu"
    sys.modules['comfy'].model_management.intermediate_dtype.return_value = torch.bfloat16
    sys.modules['comfy.model_management'].intermediate_device.return_value = "cpu"
    sys.modules['comfy.model_management'].intermediate_dtype.return_value = torch.bfloat16

    # 2. Setup workload
    raw_workload = {
      "groups": [
        {
          "group_name": "lib_group",
          "lora": {
            "name": "lib_lora",
            "urn": "urn:air:flux1:lora:civitai:7777@8888",
            "url": "https://example.com/lib_lora.safetensors"
          },
          "prompts": [
            {
              "name": "lib_prompt_1",
              "pos": "A ComfyUI library prompt"
            }
          ]
        }
      ]
    }

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
    pipeline = FluxComfyLibPipeline()

    # Configure comfy.utils mock return value directly
    sys.modules['comfy.utils'].load_torch_file.return_value = ({}, {})
    sys.modules['comfy'].utils.load_torch_file.return_value = ({}, {})
    runner.run(pipeline=pipeline, raw_workload=raw_workload, config=config)

    # 3. Verify results
    expected_image = outputs_dir / "lib_group" / "lib_prompt_1.png"
    expected_meta = outputs_dir / "lib_group" / "lib_prompt_1.json"

    assert expected_image.exists(), "Image was not saved by runner"
    assert expected_meta.exists(), "Metadata was not saved"
    
    state_mgr = SQLiteStateManager(str(db_path))
    assert state_mgr.is_completed("lib_group___lib_prompt_1") == True
