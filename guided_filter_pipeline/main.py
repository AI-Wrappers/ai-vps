import os
import logging
import sys

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

from guided_filter_pipeline.schemas import PipelineConfig
from guided_filter_pipeline.processor import MaskPairWorkloadProcessor
from guided_filter_pipeline.saver import GuidedFilterResultSaver
from guided_filter_pipeline.pipeline import GuidedFilterPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    if len(sys.argv) < 3:
        logging.error(
            "Usage: python -m guided_filter_pipeline.main <path_to_1k_masks> [path_to_4k_images] <path_to_dst_directory>\n"
            "Or: python -m guided_filter_pipeline.main <path_to_in_directory> <path_to_out_directory>"
        )
        sys.exit(1)

    if len(sys.argv) == 3:
        src_masks = sys.argv[1]
        src_images = sys.argv[1]
        dst_dir = sys.argv[2]
    else:
        src_masks = sys.argv[1]
        src_images = sys.argv[2]
        dst_dir = sys.argv[3]

    if not os.path.exists(src_masks) or not os.path.exists(src_images):
        logging.error(f"Source directories not found.")
        sys.exit(1)

    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    vram_limit_pct = float(os.environ.get("VRAM_LIMIT_PCT", "1.0"))

    merged_workload = {
        "src_masks": src_masks,
        "src_images": src_images,
        "dst": dst_dir,
        "batch_size": batch_size
    }

    runner = Runner(
        workload_processor=MaskPairWorkloadProcessor(default_batch_size=batch_size),
        state_manager=SQLiteStateManager("data/guided_filter_state_v2.db"),
        fetcher=ModelFetcher(
            "data/models_cache",
            tokens_for_provider={
                Provider.CIVITAI: os.environ.get("CIVITAI_API_KEY"),
                Provider.HUGGINGFACE: os.environ.get("HF_TOKEN"),
            },
        ),
        loop_manager=LoopManager(),
        result_saver=GuidedFilterResultSaver(),
    )

    config = PipelineConfig(vram_limit_pct=vram_limit_pct)
    pipeline = GuidedFilterPipeline()

    logging.info(f"Running GuidedFilterPipeline with batch_size={batch_size}")
    runner.run(pipeline=pipeline, raw_workload=merged_workload, config=config)

if __name__ == "__main__":
    main()
