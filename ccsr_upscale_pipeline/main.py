import os
import logging
import sys
import json

# Ensure the framework is in the python path if needed (for local tests)
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ai-pipeline-toolbox/src")
    ),
)

from ai_pipeline_toolbox.orchestrator.runner import Runner
from ai_pipeline_toolbox.components.state_manager import SQLiteStateManager
from ai_pipeline_toolbox.components.model_fetcher import ModelFetcher
from ai_pipeline_toolbox.components.loop_manager import LoopManager
from ai_pipeline_toolbox.registry.generated_enums import Provider

from ccsr_upscale_pipeline.schemas import PipelineConfig
from ccsr_upscale_pipeline.processor import DirectoryPromptWorkloadProcessor
from ccsr_upscale_pipeline.saver import CcsrUpscaleResultSaver
from ccsr_upscale_pipeline.pipeline import CCSRUpscalePipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    if len(sys.argv) < 3:
        logging.error(
            "Usage: python -m ccsr_upscale_pipeline.main <path_to_flux_workload.json> <gdrive_src_folder_id> [gdrive_dst_folder_id]"
        )
        sys.exit(1)

    flux_workload_path = sys.argv[1]
    src_dir = sys.argv[2]
    dst_dir = sys.argv[3] if len(sys.argv) > 3 else "data/outputs_ccsr"

    if not os.path.exists(flux_workload_path):
        logging.error(f"Flux workload file not found: {flux_workload_path}")
        sys.exit(1)

    with open(flux_workload_path, "r", encoding="utf-8") as f:
        flux_workload = json.load(f)

    scale_factor = 4

    merged_workload = {
        "flux_workload": flux_workload,
        "src": src_dir,
        "dst": dst_dir,
    }

    runner = Runner(
        workload_processor=DirectoryPromptWorkloadProcessor(),
        state_manager=SQLiteStateManager("data/ccsr_upscale_state.db"),
        fetcher=ModelFetcher(
            "data/models_cache",
            tokens_for_provider={
                Provider.CIVITAI: os.environ.get("CIVITAI_API_KEY"),
                Provider.HUGGINGFACE: os.environ.get("HF_TOKEN"),
            },
        ),
        loop_manager=LoopManager(),
        result_saver=CcsrUpscaleResultSaver(),
    )

    import random

    config = PipelineConfig(
        seed=random.randint(0, 999999),
        scale_factor=scale_factor,
    )
    pipeline = CCSRUpscalePipeline()

    logging.info("Running CCSRUpscalePipeline (single-image with dynamic tiling)")
    runner.run(pipeline=pipeline, raw_workload=merged_workload, config=config)


if __name__ == "__main__":
    main()
