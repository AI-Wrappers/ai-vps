import logging
import sys
import os

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
    raw_workload = {
      "groups": [
        {
          "group_name": "01_fantasy_autumn_cat",
          "lora": {
            "name": "style_filter_xu_er_thick_paint",
            "urn": "urn:air:flux1:lora:civitai:768917@1138415",
            "url": "https://civitai.com/models/768917/style-filter-xu-er-thick-paint-composition-light-texture-enhancement?modelVersionId=1138415",
            "trigger_words": [
                "XUER guangying"
            ]
        },
          "prompts": [
            {
              "name": "01_01_fluffy_tabby_lantern",
              "pos": "An ultra-detailed dark fantasy digital painting of a majestic, fluffy orange tabby cat next to a bronze lantern"
            },
            {
              "name": "01_02_special_cyber_cat",
              "pos": "An ultra-detailed cyberpunk style cat"
            }
          ]
        }
      ]
    }
    
    # Initialize framework components
    runner = Runner(
        workload_processor=GroupedWorkloadProcessor(),
        state_manager=SQLiteStateManager("data/state.db"),
        fetcher=ModelFetcher("data/models_cache"),
        loop_manager=LoopManager(),
        result_saver=ImageGroupResultSaver("data/outputs")
    )
    
    # Run with 20 steps for faster execution
    config = FluxConfig(num_inference_steps=20, guidance_scale=3.5)

    pipeline = Flux1DPipeline()
    
    runner.run(pipeline=pipeline, raw_workload=raw_workload, config=config)

if __name__ == "__main__":
    main()
