import os
import logging
import sys
import json

# Ensure the framework is in the python path if needed (for local tests)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ai-pipeline-toolbox-repo/src')))

from ai_pipeline_toolbox.orchestrator.runner import Runner
from ai_pipeline_toolbox.components.state_manager import SQLiteStateManager
from ai_pipeline_toolbox.components.model_fetcher import ModelFetcher
from ai_pipeline_toolbox.components.loop_manager import LoopManager
from ai_pipeline_toolbox.registry.generated_enums import Provider

from flux_hf_pipeline.schemas import FluxConfig
from flux_hf_pipeline.processor import GroupedWorkloadProcessor
from flux_hf_pipeline.saver import ImageGroupResultSaver
from flux_comfy_lib_pipeline.pipeline import FluxComfyLibPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    if len(sys.argv) < 2:
        logging.error("Usage: python -m flux_comfy_lib_pipeline.main <path_to_workload.json>")
        sys.exit(1)

    workload_path = sys.argv[1]
    if not os.path.exists(workload_path):
        logging.error(f"Workload file not found: {workload_path}")
        sys.exit(1)

    with open(workload_path, "r", encoding="utf-8") as f:
        raw_workload = json.load(f)

    # Initialize framework components
    runner = Runner(
        workload_processor=GroupedWorkloadProcessor(),
        state_manager=SQLiteStateManager("data/state.db"),
        fetcher=ModelFetcher(
            "data/models_cache",
            tokens_for_provider={
                Provider.CIVITAI: os.environ.get("CIVITAI_API_KEY"),
                Provider.HUGGINGFACE: os.environ.get("HF_TOKEN")
            }
        ),
        loop_manager=LoopManager(),
        result_saver=ImageGroupResultSaver("data/outputs")
    )

    config = FluxConfig(num_inference_steps=28, guidance_scale=3.5)

    pipeline = FluxComfyLibPipeline()
    
    runner.run(pipeline=pipeline, raw_workload=raw_workload, config=config)

if __name__ == "__main__":
    main()
