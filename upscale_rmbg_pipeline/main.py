import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import logging
import sys
import json

# Ensure the framework is in the python path if needed (for local tests)
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../ai-pipeline-toolbox-repo/src")
    ),
)

from ai_pipeline_toolbox.orchestrator.runner import Runner
from ai_pipeline_toolbox.components.state_manager import SQLiteStateManager
from ai_pipeline_toolbox.components.model_fetcher import ModelFetcher
from ai_pipeline_toolbox.components.loop_manager import LoopManager
from ai_pipeline_toolbox.registry.generated_enums import Provider

from upscale_rmbg_pipeline.schemas import PipelineConfig
from upscale_rmbg_pipeline.processor import DirectoryWorkloadProcessor
from upscale_rmbg_pipeline.saver import MultiArtifactSaver
from upscale_rmbg_pipeline.pipeline import BgRemovalUpscalePipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    if len(sys.argv) < 3:
        logging.error(
            "Usage: python -m upscale_rmbg_pipeline.main <path_to_flux_workload.json> <path_to_src_directory>"
        )
        sys.exit(1)

    flux_workload_path = sys.argv[1]
    src_dir = sys.argv[2]

    if not os.path.exists(flux_workload_path):
        logging.error(f"Flux workload file not found: {flux_workload_path}")
        sys.exit(1)
        
    if not os.path.exists(src_dir):
        logging.error(f"Source directory not found: {src_dir}")
        sys.exit(1)

    with open(flux_workload_path, "r", encoding="utf-8") as f:
        flux_workload = json.load(f)

    # Hardcoded or default configurations declared directly in main
    dst_dir = "data/outputs"
    scale_factor = 4
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    vram_limit_pct = 1.0

    # Merge inputs to be handled by the runner and DirectoryWorkloadProcessor
    merged_workload = {
        "flux_workload": flux_workload,
        "src": src_dir,
        "dst": dst_dir,
        "scale_factor": scale_factor,
        "batch_size": batch_size
    }

    # Initialize framework components
    runner = Runner(
        workload_processor=DirectoryWorkloadProcessor(default_batch_size=batch_size),
        state_manager=SQLiteStateManager("data/state.db"),
        fetcher=ModelFetcher(
            "data/models_cache",
            tokens_for_provider={
                Provider.CIVITAI: os.environ.get("CIVITAI_API_KEY"),
                Provider.HUGGINGFACE: os.environ.get("HF_TOKEN"),
            },
        ),
        loop_manager=LoopManager(),
        result_saver=MultiArtifactSaver(),
    )

    config = PipelineConfig(vram_limit_pct=vram_limit_pct, scale_factor=scale_factor)

    pipeline = BgRemovalUpscalePipeline()

    logger.info(
        f"Running pipeline with batch_size={batch_size}, vram_limit_pct={vram_limit_pct}, scale_factor={scale_factor}"
    )
    runner.run(pipeline=pipeline, raw_workload=merged_workload, config=config)

if __name__ == "__main__":
    main()
