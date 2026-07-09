import json
from typing import Any, Iterable
from ai_pipeline_toolbox.core.interfaces import BaseWorkloadProcessor
from flux_hf_pipeline.schemas import LoraInfo, FluxTask

class GroupedWorkloadProcessor(BaseWorkloadProcessor):
    """
    Unrolls hierarchical 'groups' JSON structure into flat FluxTask objects.
    """
    def process(self, raw_workload: Any) -> Iterable[FluxTask]:
        if isinstance(raw_workload, str):
            try:
                raw_workload = json.loads(raw_workload)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}")

        if not isinstance(raw_workload, dict):
            raise ValueError("Workload must be a dictionary containing 'groups'.")

        groups = raw_workload.get("groups", [])
        validated_tasks = []
        for group in groups:
            group_name = group.get("group_name", "default_group")
            lora_data = group.get("lora")
            lora = LoraInfo(**lora_data) if lora_data else None
            
            for prompt_data in group.get("prompts", []):
                name = prompt_data.get("name", "unnamed")
                prompt = prompt_data.get("pos", "")
                task_id = f"{group_name}___{name}"
                
                task = FluxTask(
                    task_id=task_id,
                    group_name=group_name,
                    name=name,
                    prompt=prompt,
                    lora=lora
                )
                validated_tasks.append(task)
                
        return validated_tasks
