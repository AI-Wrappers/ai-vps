import os
import logging
import sys

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

from bg_removal_pipeline.schemas import PipelineConfig
from bg_removal_pipeline.processor import SimpleDirectoryWorkloadProcessor
from bg_removal_pipeline.saver import BgRemovalResultSaver
from bg_removal_pipeline.pipeline import BgRemovalPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    if len(sys.argv) < 3:
        logging.error(
            "Usage: python -m bg_removal_pipeline.main <path_to_src_directory> <path_to_dst_directory>"
        )
        sys.exit(1)

    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]

    if not os.path.exists(src_dir):
        logging.error(f"Source directory not found: {src_dir}")
        sys.exit(1)

    batch_size = 4
    vram_limit_pct = 1.0

    merged_workload = {
        "src": src_dir,
        "dst": dst_dir,
        "batch_size": batch_size
    }

    runner = Runner(
        workload_processor=SimpleDirectoryWorkloadProcessor(default_batch_size=batch_size),
        state_manager=SQLiteStateManager("data/bg_removal_state.db"),
        fetcher=ModelFetcher(
            "data/models_cache",
            tokens_for_provider={
                Provider.CIVITAI: os.environ.get("CIVITAI_API_KEY"),
                Provider.HUGGINGFACE: os.environ.get("HF_TOKEN"),
            },
        ),
        loop_manager=LoopManager(),
        result_saver=BgRemovalResultSaver(),
    )

    config = PipelineConfig(vram_limit_pct=vram_limit_pct)
    pipeline = BgRemovalPipeline()

    logging.info(f"Running BgRemovalPipeline with batch_size={batch_size}")
    runner.run(pipeline=pipeline, raw_workload=merged_workload, config=config)

if __name__ == "__main__":
    main()
