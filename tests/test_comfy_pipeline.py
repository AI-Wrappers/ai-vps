import os
import pytest
import json
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
from flux_comfy_pipeline.pipeline import FluxComfyPipeline

@patch('flux_comfy_pipeline.pipeline.websocket.WebSocket')
@patch('flux_comfy_pipeline.pipeline.urllib.request.urlopen')
@patch('flux_comfy_pipeline.pipeline.subprocess.Popen')
def test_comfy_pipeline_flow(mock_popen, mock_urlopen, mock_websocket, tmp_path):
    # 1. Mock fetcher
    def mock_fetch_func(models):
        return {m: "/dummy/path/model.safetensors" for m in models}
    
    mock_fetcher = MagicMock(spec=ModelFetcher)
    mock_fetcher.fetch.side_effect = mock_fetch_func

    # Mock subprocess.Popen
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc

    # 2. Mock urllib.request.urlopen for different requests
    mock_resp_readiness = MagicMock()
    mock_resp_readiness.__enter__.return_value = mock_resp_readiness
    mock_resp_readiness.status = 200
    mock_resp_readiness.read.return_value = b"OK"

    mock_resp_prompt = MagicMock()
    mock_resp_prompt.__enter__.return_value = mock_resp_prompt
    mock_resp_prompt.read.return_value = json.dumps({"prompt_id": "mock_prompt_id"}).encode('utf-8')

    mock_resp_history = MagicMock()
    mock_resp_history.__enter__.return_value = mock_resp_history
    mock_resp_history.read.return_value = json.dumps({
        "mock_prompt_id": {
            "outputs": {
                "21": {
                    "images": [
                        {"filename": "test_output.png", "subfolder": "", "type": "output"}
                    ]
                }
            }
        }
    }).encode('utf-8')

    def mock_urlopen_func(req, *args, **kwargs):
        url = req.full_url if hasattr(req, 'full_url') else str(req)
        if "history" in url:
            return mock_resp_history
        elif "prompt" in url:
            return mock_resp_prompt
        else:
            return mock_resp_readiness

    mock_urlopen.side_effect = mock_urlopen_func

    # 3. Mock websocket messages
    mock_ws = MagicMock()
    mock_ws.recv.side_effect = [
        json.dumps({"type": "status", "data": {}}),
        json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "mock_prompt_id"}})
    ]
    mock_websocket.return_value = mock_ws

    # 4. Mock the saved image on disk
    dummy_image = Image.new('RGB', (64, 64), color='green')
    comfy_out_dir = Path("/workspace/ComfyUI/output")
    comfy_out_dir.mkdir(parents=True, exist_ok=True)
    dummy_image.save(comfy_out_dir / "test_output.png")

    # 5. Setup test directories
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
    pipeline = FluxComfyPipeline()

    raw_workload = {
      "groups": [
        {
          "group_name": "comfy_group",
          "lora": {
            "name": "comfy_lora",
            "urn": "urn:air:flux1:lora:civitai:9999@1111",
            "url": "https://example.com/comfy_lora.safetensors"
          },
          "prompts": [
            {
              "name": "comfy_prompt_1",
              "pos": "A ComfyUI prompt"
            }
          ]
        }
      ]
    }

    # Patch symlink function to avoid os.symlink errors on non-existent source paths
    with patch('flux_comfy_pipeline.pipeline.safe_symlink') as mock_symlink:
        runner.run(pipeline=pipeline, raw_workload=raw_workload, config=config)

    # 6. Verify assertions
    expected_image = outputs_dir / "comfy_group" / "comfy_prompt_1.png"
    expected_meta = outputs_dir / "comfy_group" / "comfy_prompt_1.json"

    assert expected_image.exists(), "Image was not saved by runner"
    assert expected_meta.exists(), "Metadata was not saved"

    # Clean up mock file
    if (comfy_out_dir / "test_output.png").exists():
        (comfy_out_dir / "test_output.png").unlink()
