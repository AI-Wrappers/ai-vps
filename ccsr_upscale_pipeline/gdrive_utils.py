import os
import io
import json
import logging
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from tenacity import retry, wait_exponential, stop_after_attempt

logger = logging.getLogger(__name__)

class GDriveClient:
    def __init__(self):
        creds_json_str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not creds_json_str:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not set.")
        try:
            creds_info = json.loads(creds_json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
            
        self.credentials = service_account.Credentials.from_service_account_info(
            creds_info, scopes=["https://www.googleapis.com/auth/drive"]
        )
        self.service = build('drive', 'v3', credentials=self.credentials, cache_discovery=False)
        
    def extract_id(self, folder_str: str) -> str:
        if "id=" in folder_str:
            return folder_str.split("id=")[1].split("&")[0]
        if "/d/" in folder_str:
            return folder_str.split("/d/")[1].split("/")[0]
        return folder_str

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
    def list_files_recursively(self, folder_id: str, current_path: Path = Path("")) -> list[dict]:
        """Returns a list of dicts: {'id': file_id, 'name': name, 'rel_path': rel_path_obj, 'parent_id': parent_id}"""
        folder_id = self.extract_id(folder_id)
        results = []
        page_token = None
        while True:
            response = self.service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType)',
                pageToken=page_token
            ).execute()
            
            for file in response.get('files', []):
                rel_path = current_path / file['name']
                if file['mimeType'] == 'application/vnd.google-apps.folder':
                    results.extend(self.list_files_recursively(file['id'], rel_path))
                else:
                    results.append({'id': file['id'], 'name': file['name'], 'rel_path': rel_path, 'parent_id': folder_id})
            
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
        return results

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
    def download_file(self, file_id: str, local_path: Path) -> None:
        logger.info(f"Downloading Google Drive file {file_id} to {local_path}")
        request = self.service.files().get_media(fileId=file_id)
        with open(local_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
    def upload_file(self, parent_id: str, file_name: str, file_stream: io.BytesIO, mime_type: str = "image/png") -> str:
        logger.info(f"Uploading file {file_name} to Google Drive folder {parent_id}")
        file_metadata = {
            'name': file_name,
            'parents': [parent_id]
        }
        media = MediaIoBaseUpload(file_stream, mimetype=mime_type, resumable=True)
        file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')
