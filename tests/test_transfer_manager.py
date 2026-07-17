import pytest
import time
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
from PIL import Image

from ccsr_upscale_pipeline.gdrive_utils import GDriveTransferManager, GDriveDownloader
from ccsr_upscale_pipeline.saver import CcsrUpscaleResultSaver
from ccsr_upscale_pipeline.schemas import ImageItem, SingleTask

@pytest.fixture(autouse=True)
def clean_singletons():
    # Reset GDriveTransferManager and GDriveDownloader singleton state before each test
    GDriveTransferManager._instance = None
    GDriveTransferManager._initialized = False
    GDriveDownloader._instance = None
    GDriveDownloader._initialized = False

def test_transfer_manager_singleton():
    manager1 = GDriveTransferManager()
    manager2 = GDriveTransferManager()
    assert manager1 is manager2

@patch("ccsr_upscale_pipeline.gdrive_utils.GDriveClient")
def test_transfer_manager_upload_backlog_flow(mock_gdrive_cls, tmp_path):
    mock_gdrive = MagicMock()
    mock_gdrive.upload_file.return_value = "drive_file_999"
    mock_gdrive_cls.return_value = mock_gdrive

    dst_root = tmp_path / "dst"
    dst_root.mkdir()

    manager = GDriveTransferManager(dst_root=str(dst_root))
    manager.gdrive = mock_gdrive
    
    # We will simulate saver.py saving a result
    # We need a SingleTask / ImageItem
    item = ImageItem(
        input_path=str(tmp_path / "temp_in.png"),
        relative_path="group_test/img_test.png",
        prompt="A test image",
        negative_prompt="",
        parent_id="folder_abc_123",
        file_id="original_file_id"
    )
    
    result = {
        "dst_root": str(dst_root),
        "items": {
            "group_test/img_test.png": {
                "item": item,
                "upscale_4k": Image.new("RGB", (10, 10), color="green")
            }
        }
    }

    # Create dummy temp input file so saver can delete it
    temp_in = Path(item.input_path)
    temp_in.parent.mkdir(parents=True, exist_ok=True)
    temp_in.touch()

    # Call save on CcsrUpscaleResultSaver
    saver = CcsrUpscaleResultSaver()
    saver.save(result, None)

    # Check local PNG and JSON exist
    expected_png = dst_root / "group_test" / "img_test_upscaled.png"
    expected_json = dst_root / "group_test" / "img_test_upscaled.json"
    
    assert expected_png.exists()
    assert expected_json.exists()
    assert not temp_in.exists() # Input was cleaned up

    # The background worker thread should scan and submit upload
    # Let's wait a moment for the background loop to process the upload
    # Or we can trigger scan manually to make sure
    manager.trigger_scan()
    
    # Wait for background queue to upload and delete files
    retries = 30
    while retries > 0 and (expected_png.exists() or expected_json.exists()):
        time.sleep(0.1)
        retries -= 1

    assert not expected_png.exists(), "PNG file should be deleted on successful upload"
    assert not expected_json.exists(), "JSON metadata file should be deleted on successful upload"
    mock_gdrive.upload_file.assert_called_once()
    
    manager.stop()

@patch("ccsr_upscale_pipeline.gdrive_utils.GDriveClient")
def test_transfer_manager_backlog_recovery_on_startup(mock_gdrive_cls, tmp_path):
    mock_gdrive = MagicMock()
    mock_gdrive_cls.return_value = mock_gdrive

    dst_root = tmp_path / "dst"
    dst_root.mkdir()

    # Pre-create leftover files to simulate crash recovery
    leftover_dir = dst_root / "group_leftover"
    leftover_dir.mkdir(parents=True, exist_ok=True)
    
    leftover_png = leftover_dir / "leftover_upscaled.png"
    leftover_json = leftover_dir / "leftover_upscaled.json"
    
    Image.new("RGB", (10, 10), color="blue").save(leftover_png)
    metadata = {"parent_id": "target_folder_99", "relative_path": "group_leftover/leftover.png"}
    with open(leftover_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    # Start the manager. It should scan dst_root on startup and process files in FIFO
    manager = GDriveTransferManager(dst_root=str(dst_root))
    manager.gdrive = mock_gdrive

    # Wait for background queue to recover, upload and delete files
    retries = 30
    while retries > 0 and (leftover_png.exists() or leftover_json.exists()):
        time.sleep(0.1)
        retries -= 1

    assert not leftover_png.exists()
    assert not leftover_json.exists()
    mock_gdrive.upload_file.assert_called_once_with("target_folder_99", "leftover_upscaled.png", ANY)

    manager.stop()
