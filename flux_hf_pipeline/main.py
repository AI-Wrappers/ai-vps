import logging
import sys
import os
import json


# Ensure the framework is in the python path if needed (for local tests)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ai-pipeline-toolbox-repo/src')))

from ai_pipeline_toolbox.orchestrator.runner import Runner
from ai_pipeline_toolbox.components.state_manager import SQLiteStateManager
from ai_pipeline_toolbox.components.model_fetcher import ModelFetcher
from ai_pipeline_toolbox.components.loop_manager import LoopManager

from flux_hf_pipeline.schemas import FluxConfig
from flux_hf_pipeline.processor import GroupedWorkloadProcessor
from flux_hf_pipeline.saver import ImageGroupResultSaver
from flux_hf_pipeline.pipeline import Flux1DPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    if len(sys.argv) < 2:
        logging.error("Usage: python -m flux_hf_pipeline.main <path_to_workload.json>")
        sys.exit(1)

    workload_path = sys.argv[1]
    if not os.path.exists(workload_path):
        logging.error(f"Workload file not found: {workload_path}")
        sys.exit(1)

    with open(workload_path, "r", encoding="utf-8") as f:
        raw_workload = json.load(f)

    hf_token = os.getenv("HF_TOKEN")
    civitai_token = os.getenv("CIVITAI_API_KEY")

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Initialize framework components
    runner = Runner(
        workload_processor=GroupedWorkloadProcessor(),
        state_manager=SQLiteStateManager("data/state.db"),
        fetcher=ModelFetcher("data/models_cache", hf_token=hf_token, civitai_token=civitai_token),
        loop_manager=LoopManager(),
        result_saver=ImageGroupResultSaver("data/outputs")
    )
    

    config = FluxConfig(num_inference_steps=28, guidance_scale=3.5)

    pipeline = Flux1DPipeline()
    
    runner.run(pipeline=pipeline, raw_workload=raw_workload, config=config)

if __name__ == "__main__":
    main()
