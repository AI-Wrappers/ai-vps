import os
import sys
import uuid
import json
import time
import socket
import logging
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Union
from enum import Enum
from PIL import Image
import websocket

from ai_pipeline_toolbox.core.pipeline import BaseGenerationPipeline
from ai_pipeline_toolbox.core.models import DynamicModel
from ai_pipeline_toolbox.registry.generated_enums import Provider, Category, DiT, Vae, TextEncoders

from flux_hf_pipeline.schemas import FluxConfig, FluxTask

logger = logging.getLogger(__name__)

def safe_symlink(src: Union[str, Path], dst: Union[str, Path]) -> None:
    src_path = Path(src).resolve()
    dst_path = Path(dst)
    if dst_path.is_symlink() or dst_path.exists():
        dst_path.unlink()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.symlink_to(src_path)

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

class FluxComfyPipeline(BaseGenerationPipeline[FluxConfig, FluxTask, Image.Image]):
    required_models = [
        DiT.FLUX1D,
        Vae.FLUX1D,
        TextEncoders.CLIP_L,
        TextEncoders.CLIP_T5XXL_FP16
    ]

    def setup(self, models_paths: Dict[Union[Enum, str, DynamicModel], str]) -> None:
        logger.info("Setting up Flux ComfyUI Pipeline...")
        self.models_paths = models_paths.copy()
        
        # 1. Symlink models to ComfyUI models directory
        comfy_models_dir = Path("/workspace/ComfyUI/models")
        
        # VAE
        vae_path = models_paths[Vae.FLUX1D]
        safe_symlink(vae_path, comfy_models_dir / "vae" / "ae.safetensors")
        
        # DiT Transformer
        dit_path = models_paths[DiT.FLUX1D]
        safe_symlink(dit_path, comfy_models_dir / "diffusion_models" / "flux1-dev.safetensors")
        
        # CLIP L
        clip_l_path = models_paths[TextEncoders.CLIP_L]
        safe_symlink(clip_l_path, comfy_models_dir / "clip" / "clip_l.safetensors")
        
        # T5 XXL
        t5_path = models_paths[TextEncoders.CLIP_T5XXL_FP16]
        safe_symlink(t5_path, comfy_models_dir / "clip" / "t5xxl_fp16.safetensors")
        
        logger.info("Symlinks for base models created successfully inside ComfyUI/models/.")

        # 2. Check and start ComfyUI Server in the background
        self.comfy_port = 8188
        self.server_address = f"127.0.0.1:{self.comfy_port}"
        self.comfy_process = None

        if is_port_in_use(self.comfy_port):
            logger.info(f"Port {self.comfy_port} is already in use. Assuming ComfyUI server is already running.")
        else:
            logger.info("Starting ComfyUI server in the background...")
            log_file = Path("data/comfyui.log")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self.comfy_log = open(log_file, "w")
            
            # Start ComfyUI using python in the current virtual environment
            comfy_main = Path("/workspace/ComfyUI/main.py").resolve()
            
            env = os.environ.copy()
            python_path = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"/workspace/ComfyUI{os.pathsep}{python_path}" if python_path else "/workspace/ComfyUI"
            
            self.comfy_process = subprocess.Popen(
                [
                    sys.executable,
                    str(comfy_main),
                    "--port", str(self.comfy_port),
                ],
                stdout=self.comfy_log,
                stderr=self.comfy_log,
                cwd="/workspace/ComfyUI",
                env=env
            )
            
            # Wait for server to be ready
            ready = False
            for i in range(30):
                time.sleep(2)
                try:
                    req = urllib.request.Request(f"http://{self.server_address}/")
                    with urllib.request.urlopen(req) as resp:
                        if resp.status == 200:
                            ready = True
                            break
                except Exception:
                    pass
            
            if not ready:
                raise RuntimeError("ComfyUI server failed to start or respond on port 8188 after 60 seconds.")
            logger.info("ComfyUI server is ready and responding.")

        self.client_id = str(uuid.uuid4())
        self.active_lora_urn = None

    def get_dynamic_models(self, workload: FluxTask) -> List[DynamicModel]:
        if workload.lora:
            return [DynamicModel(
                name=workload.lora.name,
                urn=workload.lora.urn,
                url=workload.lora.url,
                category="Lora"
            )]
        return []

    def __call__(self, config: FluxConfig, workload: FluxTask) -> Image.Image:
        logger.info(f"Handling workload: {workload.task_id}")
        
        # 1. Symlink LoRA if present
        lora_filename = None
        if workload.lora:
            lora_model = self.get_dynamic_models(workload)[0]
            # Retrieve the local path from models_paths (provided by the runner)
            # If not in models_paths directly, check self.models_paths
            lora_path = self.models_paths.get(lora_model) or self.models_paths.get(workload.lora.urn)
            if not lora_path:
                # Runner matches by Enum, string, or DynamicModel
                for k, v in self.models_paths.items():
                    if getattr(k, 'urn', None) == workload.lora.urn or str(k) == workload.lora.urn:
                        lora_path = v
                        break
            
            if not lora_path:
                logger.error(f"LoRA path not found in models_paths for URN: {workload.lora.urn}")
                raise FileNotFoundError(f"LoRA path not found in models_paths")

            # Symlink the LoRA into ComfyUI's loras folder
            # Ensure name has .safetensors extension
            lora_filename = f"{workload.lora.name}.safetensors"
            if not lora_filename.endswith(".safetensors"):
                lora_filename += ".safetensors"
                
            # If the clean name version of the file exists in the directory, prefer it over the fetched HTML/temporary path!
            clean_path = Path(lora_path).parent / lora_filename
            if clean_path.exists():
                lora_path = str(clean_path)
                
            comfy_loras_dir = Path("/workspace/ComfyUI/models/loras")
            safe_symlink(lora_path, comfy_loras_dir / lora_filename)
            logger.info(f"Symlinked LoRA {workload.lora.name} to {comfy_loras_dir / lora_filename}")

        # 2. Build the ComfyUI API workflow dictionary
        # We start with the base workflow JSON provided by the user, but adapt it dynamically.
        task_id = workload.task_id
        group_name = task_id.split("___")[0] if "___" in task_id else "unknown"
        task_name = task_id.split("___")[1] if "___" in task_id else task_id
        output_folder = f"/workspace/ComfyUI/output/{group_name}"

        workflow = {
            "4": {
                "inputs": {
                    "unet_name": "flux1-dev.safetensors",
                    "weight_dtype": "default"
                },
                "class_type": "UNETLoader"
            },
            "5": {
                "inputs": {
                    "samples": ["29", 0],
                    "vae": ["6", 0]
                },
                "class_type": "VAEDecode"
            },
            "6": {
                "inputs": {
                    "vae_name": "ae.safetensors"
                },
                "class_type": "VAELoader"
            },
            "7": {
                "inputs": {
                    "clip_l": workload.prompt,
                    "t5xxl": workload.prompt,
                    "guidance": config.guidance_scale,
                    "clip": ["52", 1] if workload.lora else ["8", 0]
                },
                "class_type": "CLIPTextEncodeFlux"
            },
            "8": {
                "inputs": {
                    "clip_name1": "clip_l.safetensors",
                    "clip_name2": "t5xxl_fp16.safetensors",
                    "type": "flux",
                    "device": "default"
                },
                "class_type": "DualCLIPLoader"
            },
            "18": {
                "inputs": {
                    "scheduler": "normal",
                    "steps": config.num_inference_steps,
                    "denoise": 1.0,
                    "model": ["25", 0]
                },
                "class_type": "BasicScheduler"
            },
            "20": {
                "inputs": {
                    "sampler_name": "euler"
                },
                "class_type": "KSamplerSelect"
            },
            "21": {
                "inputs": {
                    "filename_prefix": task_name,
                    "output_folder": output_folder,
                    "images": ["5", 0]
                },
                "class_type": "SaveImageKJ"
            },
            "25": {
                "inputs": {
                    "max_shift": 1.15,
                    "base_shift": 0.5,
                    "width": config.width,
                    "height": config.height,
                    "model": ["52", 0] if workload.lora else ["4", 0]
                },
                "class_type": "ModelSamplingFlux"
            },
            "28": {
                "inputs": {
                    "width": config.width,
                    "height": config.height,
                    "batch_size": 1
                },
                "class_type": "EmptySD3LatentImage"
            },
            "29": {
                "inputs": {
                    "noise": ["31", 0],
                    "guider": ["30", 0],
                    "sampler": ["20", 0],
                    "sigmas": ["18", 0],
                    "latent_image": ["28", 0]
                },
                "class_type": "SamplerCustomAdvanced"
            },
            "30": {
                "inputs": {
                    "model": ["52", 0] if workload.lora else ["4", 0],
                    "conditioning": ["7", 0]
                },
                "class_type": "BasicGuider"
            },
            "31": {
                "inputs": {
                    "noise_seed": config.seed
                },
                "class_type": "RandomNoise"
            }
        }

        # Inject LoRA loader only if LoRA is present
        if workload.lora and lora_filename:
            workflow["52"] = {
                "inputs": {
                    "lora_name": lora_filename,
                    "strength_model": workload.lora.strength,
                    "strength_clip": workload.lora.strength,
                    "model": ["4", 0],
                    "clip": ["8", 0]
                },
                "class_type": "LoraLoader"
            }

        # 3. Queue the prompt on ComfyUI via API
        prompt_id = self._queue_prompt(workflow)
        logger.info(f"Prompt queued on ComfyUI. Prompt ID: {prompt_id}")
        
        # 4. Wait for execution to finish via WebSockets
        self._wait_for_execution(prompt_id)
        logger.info(f"Execution finished for Prompt ID: {prompt_id}")
        
        # 5. Retrieve the saved image from ComfyUI history
        history = self._get_history(prompt_id)
        node_outputs = history.get("outputs", {})
        
        # Find the SaveImageKJ node output filename
        save_node_output = node_outputs.get("21", {})
        images = save_node_output.get("images", [])
        
        filename = None
        subfolder = ""
        
        if images:
            image_info = images[0]
            filename = image_info["filename"]
            subfolder = image_info.get("subfolder", "")
        elif "strings" in save_node_output:
            # SaveImageKJ node returns the filename in the "strings" output list
            filename = save_node_output["strings"][0]
            subfolder = ""
            
        if not filename:
            # Fallback to any node output containing images
            for node_id, output in node_outputs.items():
                if "images" in output:
                    images = output["images"]
                    if images:
                        image_info = images[0]
                        filename = image_info["filename"]
                        subfolder = image_info.get("subfolder", "")
                        break
                    
        if not filename:
            raise RuntimeError(f"No output images found in ComfyUI execution history for prompt {prompt_id}")
            
        # First try to load from the custom absolute output_folder
        image_path = Path(output_folder) / filename
        if not image_path.exists():
            # Fallback to default ComfyUI output directory
            image_path = Path("/workspace/ComfyUI/output") / subfolder / filename
            
        if not image_path.exists():
            # Try loading via the ComfyUI HTTP endpoint if direct file load fails
            logger.warning(f"Local file {image_path} not found. Attempting to download from ComfyUI API...")
            image_data = self._download_image(filename, subfolder, image_info.get("type", "output"))
            import io
            img = Image.open(io.BytesIO(image_data))
        else:
            img = Image.open(image_path)
            # Load into memory
            img.load()
            
        return img

    def _queue_prompt(self, prompt: dict) -> str:
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read())
            return res["prompt_id"]

    def _wait_for_execution(self, prompt_id: str) -> None:
        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
        start_time = time.time()
        try:
            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    msg_type = message.get('type')
                    data = message.get('data', {})
                    
                    if msg_type == 'executing':
                        node_id = data.get('node')
                        p_id = data.get('prompt_id')
                        if p_id == prompt_id:
                            if node_id is None:
                                elapsed = time.time() - start_time
                                logger.info(f"ComfyUI generation completed in {elapsed:.2f} seconds.")
                                break
                            else:
                                logger.info(f"Executing ComfyUI Node: {node_id}")
                                
                    elif msg_type == 'progress':
                        p_id = data.get('prompt_id')
                        if p_id == prompt_id:
                            val = data.get('value')
                            max_val = data.get('max')
                            logger.info(f"Sampler progress: {val}/{max_val} steps")
                else:
                    continue
        finally:
            ws.close()

    def _get_history(self, prompt_id: str) -> dict:
        req = urllib.request.Request(f"http://{self.server_address}/history/{prompt_id}")
        with urllib.request.urlopen(req) as response:
            history = json.loads(response.read())
            return history[prompt_id]

    def _download_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()

    def __del__(self):
        if getattr(self, "comfy_process", None):
            logger.info("Terminating ComfyUI server...")
            self.comfy_process.terminate()
            try:
                self.comfy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.comfy_process.kill()
            if hasattr(self, "comfy_log"):
                self.comfy_log.close()
