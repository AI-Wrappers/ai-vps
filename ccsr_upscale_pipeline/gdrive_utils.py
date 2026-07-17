import os
import io
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from tenacity import retry, wait_exponential, stop_after_attempt
from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)

# Logger specifically for upload/download actions to write to .data/here.log
transfer_logger = logging.getLogger("gdrive_transfer")
transfer_logger.setLevel(logging.INFO)
transfer_logger.propagate = False
try:
    os.makedirs(".data", exist_ok=True)
    _fh = logging.FileHandler(".data/here.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    transfer_logger.addHandler(_fh)
except Exception as _e:
    logger.warning(f"Could not initialize .data/here.log FileHandler: {_e}")


def find_credentials_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(current_dir, "_put_credentials_and_token_here")):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir
    return current_dir


class GDriveClient:
    _lock = threading.Lock()

    def __init__(self, sa_creds_path=None, oauth_token_path=None):
        with GDriveClient._lock:
            creds_dir = find_credentials_dir()
            
            # 1. Initialize Read Service (Service Account)
            sa_path = sa_creds_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.path.join(creds_dir, "service_account.json")
            if not os.path.exists(sa_path):
                raise ValueError(
                    f"Service Account credentials file not found at {sa_path}. "
                    f"Please place your service_account.json file there."
                )
            try:
                with open(sa_path, "r", encoding="utf-8") as f:
                    sa_info = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to read or parse Service Account credentials from {sa_path}: {e}")
                
            self.sa_credentials = service_account.Credentials.from_service_account_info(
                sa_info, scopes=["https://www.googleapis.com/auth/drive"]
            )

            # 2. Initialize Write Service (OAuth User Token)
            token_path = oauth_token_path or os.path.join(creds_dir, "token.json")
            if not os.path.exists(token_path):
                raise ValueError(
                    f"OAuth User Token file not found at {token_path}. "
                    f"Please run authenticate_local.py first to generate it."
                )
            try:
                with open(token_path, "r", encoding="utf-8") as f:
                    token_info = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to read or parse OAuth User Token from {token_path}: {e}")

            self.oauth_credentials = Credentials.from_authorized_user_info(
                token_info, scopes=["https://www.googleapis.com/auth/drive.file"]
            )

            # Auto-refresh if expired
            if self.oauth_credentials and self.oauth_credentials.expired and self.oauth_credentials.refresh_token:
                try:
                    from google.auth.transport.requests import Request
                    self.oauth_credentials.refresh(Request())
                    with open(token_path, "w", encoding="utf-8") as f:
                        f.write(self.oauth_credentials.to_json())
                except Exception as e:
                    logger.warning(f"Failed to refresh OAuth credentials: {e}")

            self._thread_local = threading.local()

    @property
    def read_service(self):
        if not hasattr(self._thread_local, "read_service"):
            self._thread_local.read_service = build(
                "drive", "v3", credentials=self.sa_credentials, cache_discovery=False
            )
        return self._thread_local.read_service

    @property
    def write_service(self):
        if not hasattr(self._thread_local, "write_service"):
            self._thread_local.write_service = build(
                "drive", "v3", credentials=self.oauth_credentials, cache_discovery=False
            )
        return self._thread_local.write_service

    def extract_id(self, folder_str: str) -> str:
        if "id=" in folder_str:
            return folder_str.split("id=")[1].split("&")[0]
        if "/d/" in folder_str:
            return folder_str.split("/d/")[1].split("/")[0]
        return folder_str

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5)
    )
    def list_files_recursively(
        self, folder_id: str, current_path: Path = Path("")
    ) -> list[dict]:
        """Returns a list of dicts: {'id': file_id, 'name': name, 'rel_path': rel_path_obj, 'parent_id': parent_id}"""
        folder_id = self.extract_id(folder_id)
        results = []
        page_token = None
        while True:
            response = (
                self.read_service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token,
                    orderBy="name",
                )
                .execute()
            )

            for file in response.get("files", []):
                rel_path = current_path / file["name"]
                if file["mimeType"] == "application/vnd.google-apps.folder":
                    results.extend(self.list_files_recursively(file["id"], rel_path))
                else:
                    results.append(
                        {
                            "id": file["id"],
                            "name": file["name"],
                            "rel_path": rel_path,
                            "parent_id": folder_id,
                        }
                    )

            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break
        if current_path == Path(""):
            results.sort(key=lambda x: x["name"])
        return results

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5)
    )
    def download_file(self, file_id: str, local_path: Path) -> None:
        transfer_logger.info(f"Downloading Google Drive file {file_id} to {local_path}")
        request = self.read_service.files().get_media(fileId=file_id)
        with open(local_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5)
    )
    def upload_file(
        self,
        parent_id: str,
        file_name: str,
        file_stream: io.BytesIO,
        mime_type: str = "image/png",
    ) -> str:
        transfer_logger.info(f"Uploading file {file_name} to Google Drive folder {parent_id}")
        file_metadata = {"name": file_name, "parents": [parent_id]}
        media = MediaIoBaseUpload(file_stream, mimetype=mime_type, resumable=True)
        file = (
            self.write_service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        return file.get("id")

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5)
    )
    def clean_folder(self, folder_id: str) -> None:
        folder_id = self.extract_id(folder_id)
        logger.info(f"Scanning folder {folder_id} for cleanup...")
        page_token = None
        while True:
            response = (
                self.write_service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    spaces="drive",
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                )
                .execute()
            )

            for file in response.get("files", []):
                file_id = file["id"]
                file_name = file["name"]
                try:
                    logger.info(f"Permanently deleting file {file_name} ({file_id})")
                    self.write_service.files().delete(fileId=file_id).execute()
                except Exception as e:
                    logger.error(f"Failed to delete {file_name}: {e}")

            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break


class GDriveTransferManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GDriveTransferManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, dst_root: str = None):
        if getattr(self, "_initialized", False):
            if dst_root and not self.dst_root:
                self.dst_root = dst_root
                self.trigger_scan()
            return
        self.gdrive = GDriveClient()
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="gdrive_transfer")
        self.active_paths = set()
        self.active_paths_lock = threading.Lock()
        self.dst_root = dst_root
        self.wakeup_event = threading.Event()
        self.backlog_running = False
        self.backlog_thread = None
        self.backlog_lock = threading.Lock()
        self._initialized = True
        
        if dst_root:
            self.start(dst_root)

    def start(self, dst_root: str):
        with self.backlog_lock:
            self.dst_root = dst_root
            if not self.backlog_running:
                self.backlog_running = True
                self.backlog_thread = threading.Thread(target=self._backlog_worker, daemon=True, name="gdrive_backlog_worker")
                self.backlog_thread.start()
                transfer_logger.info("GDriveTransferManager backlog worker started.")

    def stop(self):
        with self.backlog_lock:
            self.backlog_running = False
            self.wakeup_event.set()
            if self.backlog_thread:
                self.backlog_thread.join(timeout=2)
                transfer_logger.info("GDriveTransferManager backlog worker stopped.")

    def trigger_scan(self):
        self.wakeup_event.set()

    def submit_download(self, file_id: str, local_path: Path, task_id: str, on_success_callback=None):
        return self.executor.submit(self._download_job, file_id, local_path, task_id, on_success_callback)

    def submit_upload(self, png_path: Path, json_path: Path):
        with self.active_paths_lock:
            if str(png_path) in self.active_paths:
                return None
            self.active_paths.add(str(png_path))
        return self.executor.submit(self._upload_job, png_path, json_path)

    def _download_job(self, file_id: str, local_path: Path, task_id: str, on_success_callback=None):
        try:
            if local_path.exists() and local_path.stat().st_size > 0:
                transfer_logger.info(f"File {local_path} already exists locally, skipping download (cached).")
                if on_success_callback:
                    on_success_callback(local_path)
                return
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.gdrive.download_file(file_id, local_path)
            if on_success_callback:
                on_success_callback(local_path)
            transfer_logger.info(f"Finished background download for task {task_id}: {local_path.name}")
        except Exception as e:
            transfer_logger.error(f"Failed background download for task {task_id}: {e}", exc_info=True)
            raise

    def _upload_job(self, png_path: Path, json_path: Path):
        try:
            if not png_path.exists() or not json_path.exists():
                transfer_logger.warning(f"Upload job skipped, missing local file(s): {png_path} or {json_path}")
                return
            
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            parent_id = meta.get("parent_id")
            if not parent_id:
                transfer_logger.error(f"Upload job skipped, missing parent_id in metadata: {json_path}")
                json_path.unlink(missing_ok=True)
                return

            transfer_logger.info(f"Uploading file {png_path.name} to Google Drive folder {parent_id}")
            
            with open(png_path, "rb") as f:
                img_byte_arr = io.BytesIO(f.read())
            
            file_id = self.gdrive.upload_file(parent_id, png_path.name, img_byte_arr)
            transfer_logger.info(f"Successfully uploaded {png_path.name} to Google Drive folder {parent_id} with ID: {file_id}")
            
            # Delete local files on success
            png_path.unlink(missing_ok=True)
            json_path.unlink(missing_ok=True)
        except Exception as e:
            transfer_logger.error(f"Failed to upload backlog file {png_path.name}: {e}")
        finally:
            with self.active_paths_lock:
                self.active_paths.discard(str(png_path))

    def _backlog_worker(self):
        import time
        while self.backlog_running:
            self.wakeup_event.clear()
            if self.dst_root:
                try:
                    dst_path = Path(self.dst_root)
                    if dst_path.exists():
                        # Find all metadata files
                        metadata_files = list(dst_path.rglob("*_upscaled.json"))
                        backlog = []
                        for json_path in metadata_files:
                            png_path = json_path.with_name(json_path.name.replace(".json", ".png"))
                            if png_path.exists():
                                mtime = json_path.stat().st_mtime
                                backlog.append((mtime, png_path, json_path))
                        
                        # Sort by mtime ascending (FIFO)
                        backlog.sort(key=lambda x: x[0])
                        
                        for _, png_path, json_path in backlog:
                            if not self.backlog_running:
                                break
                            self.submit_upload(png_path, json_path)
                except Exception as e:
                    transfer_logger.error(f"Error in backlog scan: {e}")
            
            self.wakeup_event.wait(timeout=15.0)


class GDriveDownloader:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GDriveDownloader, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, gdrive_client=None, window_size: int = 24, max_workers: int = 4):
        if getattr(self, "_initialized", False):
            return
        self.gdrive = gdrive_client or GDriveClient()
        self.window_size = window_size
        self.tasks = []  # list of SingleTask
        self.futures = {}  # task_id -> Future
        self.downloaded_files = set()  # paths that are successfully downloaded
        self.lock = threading.Lock()
        self.current_idx = 0
        self._initialized = True

    def reset(self, gdrive_client, window_size: int):
        with self.lock:
            self.gdrive = gdrive_client or GDriveClient()
            self.window_size = window_size
            self.tasks = []
            self.futures = {}
            self.downloaded_files = set()
            self.current_idx = 0

    def set_tasks(self, tasks):
        with self.lock:
            self.tasks = tasks
            # Start downloading the first window_size files
            for i in range(min(self.window_size, len(self.tasks))):
                self._start_download_nolock(i)

    def _start_download_nolock(self, index: int):
        if index >= len(self.tasks):
            return
        task = self.tasks[index]
        task_id = task.task_id
        if task_id in self.futures:
            return  # Already queued or done
            
        file_id = task.item.file_id
        local_path = Path(task.item.input_path)
        
        transfer_logger.info(f"Queueing background download for task {task_id}: {local_path.name}")
        
        def success_cb(p):
            with self.lock:
                self.downloaded_files.add(str(p))

        manager = GDriveTransferManager()
        future = manager.submit_download(file_id, local_path, task_id, on_success_callback=success_cb)
        self.futures[task_id] = future

    def wait_for_task(self, task_id: str):
        future = None
        with self.lock:
            future = self.futures.get(task_id)
            
        if future is None:
            # If not yet queued (outside current window), start it now
            with self.lock:
                # Find task index
                task_idx = -1
                for idx, t in enumerate(self.tasks):
                    if t.task_id == task_id:
                        task_idx = idx
                        break
                if task_idx != -1:
                    self._start_download_nolock(task_idx)
                    future = self.futures.get(task_id)
                    
        if future is not None:
            future.result()  # Block until complete
        else:
            raise ValueError(f"Task {task_id} not found in downloader tasks.")

    def on_task_start(self, task_idx: int):
        with self.lock:
            self.current_idx = task_idx
            # Slide window: make sure the next tasks up to task_idx + window_size - 1 are downloading
            for i in range(task_idx, min(task_idx + self.window_size, len(self.tasks))):
                self._start_download_nolock(i)
