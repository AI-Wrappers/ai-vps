import os
import io
import json
import base64
import logging
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from tenacity import retry, wait_exponential, stop_after_attempt

logger = logging.getLogger(__name__)


from google.oauth2.credentials import Credentials

class GDriveClient:
    def __init__(self, sa_creds_b64=None, oauth_token_b64=None):
        # 1. Initialize Read Service (Service Account)
        sa_b64 = sa_creds_b64 or os.environ.get("GOOGLE_GDRIVE_CREDENTIALS_BASE64") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not sa_b64:
            raise ValueError(
                "Neither GOOGLE_GDRIVE_CREDENTIALS_BASE64 nor GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable is set."
            )
        try:
            sa_json = base64.b64decode(sa_b64).decode("utf-8")
            sa_info = json.loads(sa_json)
        except Exception as e:
            raise ValueError(f"Failed to decode or parse Service Account credentials: {e}")
            
        self.sa_credentials = service_account.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/drive"]
        )
        self.read_service = build(
            "drive", "v3", credentials=self.sa_credentials, cache_discovery=False
        )

        # 2. Initialize Write Service (OAuth User Token)
        token_b64 = oauth_token_b64 or os.environ.get("GOOGLE_OAUTH_TOKEN_BASE64")
        if not token_b64:
            raise ValueError(
                "GOOGLE_OAUTH_TOKEN_BASE64 environment variable is not set."
            )
        try:
            token_json = base64.b64decode(token_b64).decode("utf-8")
            token_info = json.loads(token_json)
        except Exception as e:
            raise ValueError(f"Failed to decode or parse OAuth User Token: {e}")

        self.oauth_credentials = Credentials.from_authorized_user_info(
            token_info, scopes=["https://www.googleapis.com/auth/drive.file"]
        )

        # Auto-refresh if expired
        if self.oauth_credentials and self.oauth_credentials.expired and self.oauth_credentials.refresh_token:
            try:
                from google.auth.transport.requests import Request
                self.oauth_credentials.refresh(Request())
            except Exception as e:
                logger.warning(f"Failed to refresh OAuth credentials: {e}")

        self.write_service = build(
            "drive", "v3", credentials=self.oauth_credentials, cache_discovery=False
        )

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
        return results

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5)
    )
    def download_file(self, file_id: str, local_path: Path) -> None:
        logger.info(f"Downloading Google Drive file {file_id} to {local_path}")
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
        logger.info(f"Uploading file {file_name} to Google Drive folder {parent_id}")
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

