import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

from accelerate import Accelerator
from ai_pipeline_toolbox.orchestrator.runner import Runner
from ai_pipeline_toolbox.components.state_manager import SQLiteStateManager
from ai_pipeline_toolbox.components.model_fetcher import ModelFetcher
from ai_pipeline_toolbox.components.loop_manager import LoopManager
from ai_pipeline_toolbox.registry.generated_enums import Provider
import sqlite3

from ccsr_upscale_pipeline.schemas import PipelineConfig
from ccsr_upscale_pipeline.processor import DirectoryPromptWorkloadProcessor
from ccsr_upscale_pipeline.saver import CcsrUpscaleResultSaver
from ccsr_upscale_pipeline.pipeline import CCSRUpscalePipeline
from ccsr_upscale_pipeline.gdrive_utils import GDriveDownloader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ConcurrentSQLiteStateManager(SQLiteStateManager):
    def _init_db(self):
        # Configure connections with a long timeout to allow concurrent multi-GPU processes
        # to write task completions without locking, and enable WAL journal mode.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=60.0)
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    error TEXT
                )
            ''')


class AccelerateLoopManager(LoopManager):
    def __init__(self, accelerator: Accelerator):
        super().__init__()
        self.accelerator = accelerator

    def iterate(self, workload):
        # Shard the pending tasks across all GPU processes
        my_tasks = self.accelerator.split_between_processes(list(workload))
        
        # Re-initialize/reset the GDriveDownloader for this process so it only fetches
        # files for tasks assigned to this specific process.
        downloader = GDriveDownloader()
        downloader.reset(downloader.gdrive, window_size=24)
        downloader.set_tasks(my_tasks)
        
        logging.info(
            f"Process {self.accelerator.process_index}/{self.accelerator.num_processes} "
            f"assigned {len(my_tasks)} tasks out of {len(workload)} total pending tasks."
        )
        for task in my_tasks:
            yield task


def main():
    if len(sys.argv) < 3:
        logging.error(
            "Usage: python -m ccsr_upscale_pipeline.main <path_to_flux_workload.json> <gdrive_src_folder_id> [gdrive_dst_folder_id]"
        )
        sys.exit(1)

    accelerator = Accelerator()

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
        state_manager=ConcurrentSQLiteStateManager("data/ccsr_upscale_state.db"),
        fetcher=ModelFetcher(
            "data/models_cache",
            tokens_for_provider={
                Provider.CIVITAI: os.environ.get("CIVITAI_API_KEY"),
                Provider.HUGGINGFACE: os.environ.get("HF_TOKEN"),
            },
        ),
        loop_manager=AccelerateLoopManager(accelerator),
        result_saver=CcsrUpscaleResultSaver(),
    )

    import random

    config = PipelineConfig(
        seed=random.randint(0, 999999),
        scale_factor=scale_factor,
    )
    pipeline = CCSRUpscalePipeline(accelerator=accelerator)

    logging.info(f"Running CCSRUpscalePipeline (single-image with dynamic tiling) on process {accelerator.process_index}")
    runner.run(pipeline=pipeline, raw_workload=merged_workload, config=config)


if __name__ == "__main__":
    main()
