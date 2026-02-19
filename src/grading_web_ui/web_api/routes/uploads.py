"""
File upload and processing endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import tempfile
import zipfile
import hashlib
import os
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
import yaml

from ..models import UploadResponse
from .. import sse
from ..auth import require_instructor, require_session_access

# Store in database using repositories
from ..repositories import with_transaction
from ..domain.submission import Submission
from ..domain.problem import Problem

import logging
import asyncio
import concurrent.futures
import threading
from ..services.exam_processor import ExamProcessor, PRESCAN_DPI_STEPS, NAME_SIMILARITY_THRESHOLD
from ..services.qr_scanner import QRScanner
from lms_interface.canvas_interface import CanvasInterface
from ..repositories import SessionRepository, SubmissionRepository, ProblemMetadataRepository, ProblemRepository
from ..domain.common import SessionStatus

router = APIRouter()
log = logging.getLogger(__name__)
QUIZ_YAML_TEXT_KEY = "quiz_yaml_text"
QUIZ_YAML_FILENAME_KEY = "quiz_yaml_filename"
QUIZ_YAML_IDS_KEY = "quiz_yaml_ids"
QUIZ_YAML_DOC_COUNT_KEY = "quiz_yaml_doc_count"
QUIZ_YAML_UPLOADED_AT_KEY = "quiz_yaml_uploaded_at"


class SplitPointsSubmission(BaseModel):
  """Model for manual split points submission"""
  split_points: Dict[str, List[int]]
  skip_first_region: bool = True  # Default to skipping first region (header/title)
  last_page_blank: bool = False  # Default to not skipping last page
  ai_provider: str = "anthropic"  # AI provider for name extraction (anthropic, openai, ollama)


def compute_file_hash(file_path: Path) -> str:
  """Compute SHA256 hash of a file"""
  sha256_hash = hashlib.sha256()
  with open(file_path, "rb") as f:
    for byte_block in iter(lambda: f.read(4096), b""):
      sha256_hash.update(byte_block)
  return sha256_hash.hexdigest()


def sanitize_uploaded_filename(raw_filename: str) -> str:
  """
  Return a safe filename for local filesystem writes.

  Removes path separators and normalizes suspicious characters so uploaded
  names cannot escape the target directory.
  """
  trimmed = (raw_filename or "").strip()
  if not trimmed:
    raise HTTPException(status_code=400, detail="Uploaded file name is empty")

  normalized = trimmed.replace("\\", "/")
  candidate = Path(normalized).name.strip()
  if not candidate or candidate in (".", ".."):
    raise HTTPException(status_code=400,
                        detail=f"Invalid uploaded filename '{raw_filename}'")

  safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", candidate)
  safe_name = safe_name.strip(". ")
  if not safe_name:
    raise HTTPException(status_code=400,
                        detail=f"Invalid uploaded filename '{raw_filename}'")

  return safe_name


def extract_pdf_files_safely(zip_path: Path, extract_dir: Path) -> List[Path]:
  """
  Extract only PDF files from zip archive after traversal checks.
  """
  extracted_pdfs: List[Path] = []
  extract_root = extract_dir.resolve()

  with zipfile.ZipFile(zip_path, "r") as zip_ref:
    for member in zip_ref.infolist():
      member_name = member.filename.replace("\\", "/")
      if not member_name or member_name.endswith("/"):
        continue
      if "\x00" in member_name:
        raise HTTPException(status_code=400,
                            detail="Zip archive contains invalid file paths")

      member_path = Path(member_name)
      if member_path.is_absolute() or ".." in member_path.parts:
        raise HTTPException(status_code=400,
                            detail="Zip archive contains unsafe file paths")

      destination = (extract_root / member_path).resolve()
      if extract_root not in destination.parents:
        raise HTTPException(status_code=400,
                            detail="Zip archive contains unsafe file paths")

      if destination.suffix.lower() != ".pdf":
        continue

      destination.parent.mkdir(parents=True, exist_ok=True)
      with zip_ref.open(member, "r") as src_file, open(destination, "wb") as out_file:
        shutil.copyfileobj(src_file, out_file)
      extracted_pdfs.append(destination)

  return extracted_pdfs


def parse_uploaded_quiz_yaml(content: bytes, filename: str) -> Dict[str, Any]:
  """Validate and parse uploaded YAML content."""
  if not content:
    raise HTTPException(status_code=400,
                        detail=f"Quiz YAML file '{filename}' is empty")

  try:
    yaml_text = content.decode("utf-8")
  except UnicodeDecodeError:
    raise HTTPException(status_code=400,
                        detail=f"Quiz YAML file '{filename}' must be UTF-8 encoded")

  try:
    docs = list(yaml.safe_load_all(yaml_text))
  except yaml.YAMLError as exc:
    raise HTTPException(status_code=400,
                        detail=f"Invalid quiz YAML '{filename}': {exc}")

  yaml_ids = []
  for doc in docs:
    if isinstance(doc, dict):
      yaml_id = doc.get("yaml_id")
      if isinstance(yaml_id, str) and yaml_id.strip():
        yaml_ids.append(yaml_id.strip())

  # Preserve insertion order while removing duplicates
  seen = set()
  ordered_ids = []
  for yaml_id in yaml_ids:
    if yaml_id not in seen:
      seen.add(yaml_id)
      ordered_ids.append(yaml_id)

  return {
    "text": yaml_text,
    "filename": filename,
    "yaml_ids": ordered_ids,
    "doc_count": len(docs),
    "uploaded_at": datetime.now().isoformat(timespec="seconds")
  }


def apply_uploaded_quiz_yaml_metadata(session_data: Dict[str, Any],
                                      uploaded_yaml: Optional[Dict[str, Any]]) -> None:
  """Write parsed YAML metadata into session metadata."""
  if not uploaded_yaml:
    return
  session_data[QUIZ_YAML_TEXT_KEY] = uploaded_yaml["text"]
  session_data[QUIZ_YAML_FILENAME_KEY] = uploaded_yaml["filename"]
  session_data[QUIZ_YAML_IDS_KEY] = uploaded_yaml["yaml_ids"]
  session_data[QUIZ_YAML_DOC_COUNT_KEY] = uploaded_yaml["doc_count"]
  session_data[QUIZ_YAML_UPLOADED_AT_KEY] = uploaded_yaml["uploaded_at"]


@router.get("/{session_id}/upload-stream")
async def upload_progress_stream(
  session_id: int,
  current_user: dict = Depends(require_session_access())
):
  """SSE stream for upload/processing progress (requires session access)"""
  stream_id = sse.make_stream_id("upload", session_id)

  # Create stream if it doesn't exist
  if not sse.get_stream(stream_id):
    sse.create_stream(stream_id)

  return StreamingResponse(sse.event_generator(stream_id),
                           media_type="text/event-stream",
                           headers={
                             "Cache-Control": "no-cache",
                             "Connection": "keep-alive",
                           })


@router.post("/{session_id}/upload", response_model=UploadResponse)
async def upload_exams(
  session_id: int,
  background_tasks: BackgroundTasks,
  files: List[UploadFile] = File(...),
  current_user: dict = Depends(require_instructor)
):
  """
    Upload exam PDFs or a zip file containing exams (instructor only).
    Starts name extraction before alignment.
    """

  # Verify session exists
  session_repo = SessionRepository()
  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  # Save uploaded files temporarily and compute hashes
  temp_dir = Path(tempfile.mkdtemp())
  saved_files = []
  file_metadata = {}  # Map: file_path -> {hash, original_filename}
  filename_counter = {}  # Track filename usage to handle duplicates
  uploaded_yaml = None

  for file in files:
    # Handle duplicate filenames by appending a counter
    # This can happen when dragging folders with same filenames in different subdirectories
    original_filename = (file.filename or "").strip()
    if not original_filename:
      continue
    base_filename = sanitize_uploaded_filename(original_filename)
    if base_filename in filename_counter:
      filename_counter[base_filename] += 1
      # Insert counter before extension: "file.pdf" -> "file_1.pdf"
      stem = Path(base_filename).stem
      suffix = Path(base_filename).suffix
      unique_filename = f"{stem}_{filename_counter[base_filename]}{suffix}"
    else:
      filename_counter[base_filename] = 0
      unique_filename = base_filename

    file_path = temp_dir / unique_filename
    with open(file_path, "wb") as f:
      content = await file.read()
      f.write(content)

    suffix = file_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
      if uploaded_yaml is not None:
        raise HTTPException(
          status_code=400,
          detail="Upload exactly one quiz YAML file (.yaml/.yml) at a time")
      uploaded_yaml = parse_uploaded_quiz_yaml(content, base_filename)
      continue

    if suffix not in (".pdf", ".zip"):
      raise HTTPException(
        status_code=400,
        detail=
        f"Unsupported file type '{base_filename}'. Upload .pdf, .zip, .yaml, or .yml files.")

    # Compute hash for duplicate detection
    file_hash = compute_file_hash(file_path)
    file_metadata[file_path] = {
      "hash": file_hash,
      "original_filename": base_filename  # Store original name for display
    }

    saved_files.append(file_path)

  # If it's a zip file, extract it
  if len(saved_files) == 1 and saved_files[0].suffix == ".zip":
    zip_path = saved_files[0]
    extract_dir = temp_dir / "extracted"
    extract_dir.mkdir()

    # Find all PDFs in extracted directory and compute their hashes
    saved_files = extract_pdf_files_safely(zip_path, extract_dir)
    file_metadata = {}
    for pdf_path in saved_files:
      file_hash = compute_file_hash(pdf_path)
      file_metadata[pdf_path] = {
        "hash": file_hash,
        "original_filename": pdf_path.name
      }

  # Store file paths and metadata in session for later processing
  # Append to existing uploads if any (support multiple upload batches)

  session_repo = SessionRepository()

  # Get existing session data
  existing_data = session_repo.get_metadata(session_id) or {}

  if not saved_files:
    if uploaded_yaml:
      session_data = dict(existing_data)
      apply_uploaded_quiz_yaml_metadata(session_data, uploaded_yaml)
      session_repo.update_metadata(session_id, session_data)
      return {
        "session_id": session_id,
        "files_uploaded": 0,
        "status": "yaml_uploaded",
        "message":
        f"Uploaded quiz YAML '{uploaded_yaml['filename']}' for this session.",
        "num_exams": len(existing_data.get("file_paths", [])),
        "auto_processed": False
      }
    raise HTTPException(
      status_code=400,
      detail="No exam files found. Upload at least one PDF/ZIP exam file.")

  # Check if we have existing split points from a previous upload
  has_existing_split_points = (existing_data
                               and "split_points" in existing_data
                               and existing_data["split_points"])

  if has_existing_split_points:
    log.info(f"Found existing split points - will auto-process new uploads")
  else:
    log.info(f"No existing split points found - will show alignment UI")

  if existing_data and "file_paths" in existing_data:
    # Append to existing files
    log.info(
      f"Appending {len(saved_files)} files to existing {len(existing_data['file_paths'])} files"
    )

    existing_files = [Path(p) for p in existing_data["file_paths"]]
    existing_metadata = {
      Path(k): v
      for k, v in existing_data["file_metadata"].items()
    }

    # Combine with new files (avoiding duplicates by hash)
    existing_hashes = {meta["hash"] for meta in existing_metadata.values()}
    new_files_added = 0

    for new_file in saved_files:
      new_hash = file_metadata[new_file]["hash"]
      if new_hash not in existing_hashes:
        existing_files.append(new_file)
        existing_metadata[new_file] = file_metadata[new_file]
        new_files_added += 1
      else:
        log.info(f"Skipping duplicate file: {new_file.name}")

    log.info(
      f"Added {new_files_added} new files (skipped {len(saved_files) - new_files_added} duplicates)"
    )

    # Use the first temp_dir or create new one
    temp_dir_to_use = existing_data.get("temp_dir", str(temp_dir))

    # Preserve existing split points and settings if they exist
    session_data = {
      "temp_dir": temp_dir_to_use,
      "file_paths": [str(f) for f in existing_files],
      "file_metadata": {
        str(k): v
        for k, v in existing_metadata.items()
      },
      "mock_roster": existing_data.get("mock_roster", False),
      "ai_name_extraction": existing_data.get("ai_name_extraction", True)
    }
    apply_uploaded_quiz_yaml_metadata(session_data, uploaded_yaml)
    if not uploaded_yaml:
      # Preserve existing quiz YAML when not replaced.
      for key in (QUIZ_YAML_TEXT_KEY, QUIZ_YAML_FILENAME_KEY, QUIZ_YAML_IDS_KEY,
                  QUIZ_YAML_DOC_COUNT_KEY, QUIZ_YAML_UPLOADED_AT_KEY):
        if key in existing_data:
          session_data[key] = existing_data[key]

    # Preserve split points and other settings from previous upload
    if has_existing_split_points:
      # Convert string keys back to integers (JSON serialization converts int keys to strings)
      raw_split_points = existing_data["split_points"]
      session_data["split_points"] = {
        int(k): v
        for k, v in raw_split_points.items()
      }
      session_data["skip_first_region"] = existing_data.get(
        "skip_first_region", True)
      session_data["last_page_blank"] = existing_data.get(
        "last_page_blank", False)
      session_data["ai_provider"] = existing_data.get(
        "ai_provider", "anthropic")
      session_data["composite_dimensions"] = existing_data.get(
        "composite_dimensions", {})
      log.info(
        f"Reusing existing split points from previous upload: {session_data['split_points']}"
      )

    total_files = len(existing_files)
  else:
    # First upload for this session
    log.info(f"First upload: {len(saved_files)} files")
    session_data = {
      "temp_dir": str(temp_dir),
      "file_paths": [str(f) for f in saved_files],
      "file_metadata": {
        str(k): v
        for k, v in file_metadata.items()
      },
      "mock_roster": existing_data.get("mock_roster", False),
      "ai_name_extraction": existing_data.get("ai_name_extraction", True)
    }
    apply_uploaded_quiz_yaml_metadata(session_data, uploaded_yaml)
    if not uploaded_yaml:
      for key in (QUIZ_YAML_TEXT_KEY, QUIZ_YAML_FILENAME_KEY, QUIZ_YAML_IDS_KEY,
                  QUIZ_YAML_DOC_COUNT_KEY, QUIZ_YAML_UPLOADED_AT_KEY):
        if key in existing_data:
          session_data[key] = existing_data[key]
    total_files = len(saved_files)

  # Update session with file metadata and status
  session_repo.update_metadata(session_id, session_data)
  session_repo.update_status(
    session_id,
    SessionStatus.PREPROCESSING,
    "Uploaded. Extracting names..."
  )

  # Also update total_exams count
  session = session_repo.get_by_id(session_id)
  session.total_exams = total_files
  session_repo.update(session)
  yaml_note = ""
  if uploaded_yaml:
    yaml_note = f" Quiz YAML '{uploaded_yaml['filename']}' attached."

  # If we already have split points from a previous upload, auto-submit and skip alignment UI
  if has_existing_split_points:
    log.info(f"Auto-processing new files with existing split points")

    # Create SSE stream for progress
    stream_id = sse.make_stream_id("upload", session_id)
    sse.create_stream(stream_id)

    # Get file paths and metadata
    file_paths = [Path(p) for p in session_data["file_paths"]]
    file_metadata_dict = {
      Path(k): v
      for k, v in session_data["file_metadata"].items()
    }

    # Start background name extraction (no splitting yet)
    from fastapi import BackgroundTasks
    background_tasks = BackgroundTasks()
    background_tasks.add_task(process_exam_names, session_id, file_paths,
                              file_metadata_dict, stream_id,
                              session_data.get("ai_provider", "anthropic"))

    # Execute background tasks (they run after response is sent)
    import asyncio
    asyncio.create_task(background_tasks())

    return {
      "session_id": session_id,
      "files_uploaded": len(saved_files),
      "status": "processing",
      "message":
      f"Uploaded {len(saved_files)} exam(s). Extracting names...{yaml_note}",
      "num_exams": total_files,
      "auto_processed": True
    }

  # Start background name extraction (no splitting yet)
  file_paths = [Path(p) for p in session_data["file_paths"]]
  file_metadata_dict = {
    Path(k): v
    for k, v in session_data["file_metadata"].items()
  }

  stream_id = sse.make_stream_id("upload", session_id)
  if not sse.get_stream(stream_id):
    sse.create_stream(stream_id)

  background_tasks.add_task(
    process_exam_names,
    session_id,
    file_paths,
    file_metadata_dict,
    stream_id,
    session_data.get("ai_provider", "anthropic")
  )

  return {
    "session_id": session_id,
    "files_uploaded": len(saved_files),
    "status": "processing",
    "message":
    f"Uploaded {len(saved_files)} exam(s). Total: {total_files} exam(s). Extracting names...{yaml_note}",
    "num_exams": total_files,
    "auto_processed": False
  }


@router.post("/{session_id}/prepare-alignment", response_model=UploadResponse)
async def prepare_alignment(
  session_id: int,
  background_tasks: BackgroundTasks,
  current_user: dict = Depends(require_instructor)
):
  """
    Prepare the alignment step after name matching.
    Returns composites for manual alignment or starts auto-processing if split points exist.
    """
  from ..services.manual_alignment import ManualAlignmentService

  session_repo = SessionRepository()
  session = session_repo.get_by_id(session_id)
  if not session:
    raise HTTPException(status_code=404, detail="Session not found")

  session_data = session_repo.get_metadata(session_id)
  if not session_data or "file_paths" not in session_data:
    raise HTTPException(status_code=400, detail="No uploaded files found for this session")

  mock_roster = bool(session_data.get("mock_roster"))
  if not mock_roster:
    submission_repo = SubmissionRepository()
    if submission_repo.count_unmatched(session_id) > 0:
      raise HTTPException(status_code=400,
                          detail="Name matching is incomplete. Match all submissions before alignment.")

  file_paths = [Path(p) for p in session_data["file_paths"]]
  file_metadata = {
    Path(k): v
    for k, v in session_data["file_metadata"].items()
  }

  split_points = session_data.get("split_points")
  if split_points:
    split_points = {int(k): v for k, v in split_points.items()}
    session_data["split_points"] = split_points
    session_repo.update_metadata(session_id, session_data)
  has_existing_split_points = bool(split_points)

  if has_existing_split_points:
    stream_id = sse.make_stream_id("upload", session_id)
    sse.create_stream(stream_id)

    background_tasks.add_task(
      process_exam_splits,
      session_id,
      file_paths,
      file_metadata,
      stream_id,
      split_points,
      session_data.get("skip_first_region", True),
      session_data.get("last_page_blank", False),
      session_data.get("ai_provider", "anthropic")
    )

    session.status = SessionStatus.PREPROCESSING
    session.processed_exams = 0
    session.processing_message = "Processing with saved split points..."
    session_repo.update(session)

    return {
      "session_id": session_id,
      "files_uploaded": len(file_paths),
      "status": "processing",
      "message": "Processing exams with saved split points...",
      "num_exams": session.total_exams,
      "auto_processed": True
    }

  qr_positions_by_file = None
  qr_scanner = QRScanner()
  stream_id = sse.make_stream_id("upload", session_id)
  if not sse.get_stream(stream_id):
    sse.create_stream(stream_id)
  main_loop = asyncio.get_event_loop()

  def send_progress(message: str, processed: int, total: int) -> None:
    asyncio.run_coroutine_threadsafe(
      sse.send_event(stream_id, "progress", {
        "total": total,
        "processed": processed,
        "matched": 0,
        "progress": int((processed / total) * 100) if total else 0,
        "current_step": processed,
        "total_steps": total,
        "message": message
      }),
      main_loop
    )

  def build_alignment_assets():
    qr_positions = None
    total_qr_steps = 0
    max_pages = 0
    page_counts = []
    if qr_scanner.available:
      import fitz
      for pdf_path in file_paths:
        try:
          doc = fitz.open(str(pdf_path))
          count = doc.page_count
          doc.close()
        except Exception:
          count = 0
        page_counts.append(count)
        total_qr_steps += count
        max_pages = max(max_pages, count)

      qr_positions = {}
      total_files = len(file_paths)
      total_steps = total_qr_steps + max_pages
      if total_qr_steps > 0:
        progress_lock = threading.Lock()
        pages_scanned = {"count": 0}

        def scan_file(index: int, pdf_path: Path) -> tuple[Path, Dict[int, List[Dict]]]:
          per_file_workers = 1 if total_files > 1 else None

          def page_progress(_completed: int, _page_total: int, _message: str) -> None:
            with progress_lock:
              pages_scanned["count"] += 1
              global_completed = pages_scanned["count"]
            send_progress(
              f"Scanning QR codes ({index}/{total_files}) in {pdf_path.name} ({global_completed}/{total_qr_steps} pages)",
              global_completed,
              total_steps
            )

          positions = qr_scanner.scan_qr_positions_from_pdf(
            pdf_path,
            progress_callback=page_progress,
            max_workers=per_file_workers
          )
          return pdf_path, positions

        max_workers = min(4, total_files, os.cpu_count() or 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
          futures = [
            executor.submit(scan_file, index, pdf_path)
            for index, pdf_path in enumerate(file_paths, start=1)
          ]
          for future in concurrent.futures.as_completed(futures):
            pdf_path, positions = future.result()
            qr_positions[pdf_path] = positions
    else:
      max_pages = 1
      total_qr_steps = 0
      send_progress("Preparing alignment images...", 0, max_pages)

    alignment_service = ManualAlignmentService()
    def composite_progress(page_index: int, page_total: int, message: str) -> None:
      total_steps = total_qr_steps + page_total
      send_progress(
        message,
        total_qr_steps + page_index,
        total_steps
      )

    composites, dimensions, transforms_by_file = alignment_service.create_composite_images(
      file_paths,
      qr_positions_by_file=qr_positions,
      progress_callback=composite_progress
    )
    suggested_split_points = alignment_service.suggest_split_points_from_composites(
      composites,
      qr_positions_by_file=qr_positions,
      transforms_by_file=transforms_by_file
    )
    return composites, dimensions, transforms_by_file, suggested_split_points

  composites, dimensions, transforms_by_file, suggested_split_points = await asyncio.to_thread(build_alignment_assets)

  composite_dimensions = {
    str(page_num): [dims[0], dims[1]]
    for page_num, dims in dimensions.items()
  }
  session_data["composite_dimensions"] = composite_dimensions
  file_hash_by_path = {
    Path(k): v["hash"]
    for k, v in session_data.get("file_metadata", {}).items()
  }
  page_transforms = {}
  for pdf_path, transforms in transforms_by_file.items():
    file_hash = file_hash_by_path.get(Path(pdf_path))
    if not file_hash:
      continue
    page_transforms[file_hash] = {
      str(page_num): data for page_num, data in transforms.items()
    }
  session_data["page_transforms"] = page_transforms
  session_repo.update_metadata(session_id, session_data)
  session_repo.update_status(session_id, SessionStatus.AWAITING_ALIGNMENT,
                             "Alignment ready. Please select split points.")

  page_dimensions = {
    page_num: {"width": dims[0], "height": dims[1]}
    for page_num, dims in dimensions.items()
  }

  return {
    "session_id": session_id,
    "files_uploaded": len(file_paths),
    "status": "awaiting_alignment",
    "message": "Alignment ready. Please select split points.",
    "composites": composites,
    "page_dimensions": page_dimensions,
    "num_exams": session.total_exams,
    "suggested_split_points": suggested_split_points,
    "auto_processed": False
  }


@router.post("/{session_id}/submit-alignment")
async def submit_alignment(
  session_id: int,
  background_tasks: BackgroundTasks,
  submission: SplitPointsSubmission,
  current_user: dict = Depends(require_instructor)
):
  """
    Submit manual split points and start processing exams (instructor only).

    Args:
        session_id: Session ID
        submission: Model containing split_points dict mapping page_number (as string) -> list of y-positions
    """
  # Retrieve stored file paths from session metadata
  session_repo = SessionRepository()
  session_data = session_repo.get_metadata(session_id)

  if not session_data:
    raise HTTPException(status_code=404,
                        detail="Session not found or no files uploaded")

  # Reconstruct file paths and metadata
  file_paths = [Path(p) for p in session_data["file_paths"]]
  file_metadata = {
    Path(k): v
    for k, v in session_data["file_metadata"].items()
  }

  # Convert split_points from absolute pixels to percentages of page height
  # This makes them resolution-independent
  composite_dimensions = session_data.get("composite_dimensions", {})
  manual_split_points = {}

  for page_str, y_positions in submission.split_points.items():
    page_num = int(page_str)

    # Get composite page height for this page
    if str(page_num) in composite_dimensions:
      page_height = composite_dimensions[str(page_num)][1]  # [width, height]

      # Convert each y-position from pixels to percentage
      percentages = [y_pos / page_height for y_pos in y_positions]
      manual_split_points[page_num] = percentages
    else:
      # Fallback: if no composite dimensions, pass through as-is
      log.warning(
        f"No composite dimensions for page {page_num}, using absolute coordinates"
      )
      manual_split_points[page_num] = y_positions

  # Save split points and settings to session metadata for reuse in future uploads
  session_data["split_points"] = manual_split_points
  session_data["skip_first_region"] = submission.skip_first_region
  session_data["last_page_blank"] = submission.last_page_blank
  session_data["ai_provider"] = submission.ai_provider

  session_repo.update_metadata(session_id, session_data)

  log.info(
    f"Saved split points and settings to session metadata for future uploads")

  # Create SSE stream for progress updates
  stream_id = sse.make_stream_id("upload", session_id)
  sse.create_stream(stream_id)

  # Start background processing with manual split points
  background_tasks.add_task(
    process_exam_splits,
    session_id,
    file_paths,
    file_metadata,
    stream_id,
    manual_split_points,  # Pass manual splits
    submission.skip_first_region,  # Pass skip_first_region flag
    submission.last_page_blank,  # Pass last_page_blank flag
    submission.ai_provider  # Pass AI provider selection
  )

  # Update session status
  session = session_repo.get_by_id(session_id)
  session.status = SessionStatus.PREPROCESSING
  session.processed_exams = 0
  session.processing_message = 'Processing with manual split points...'
  session_repo.update(session)

  return {
    "session_id": session_id,
    "status": "processing",
    "message": f"Processing {len(file_paths)} exam(s) with manual alignment"
  }


async def process_exam_names(
  session_id: int,
  file_paths: List[Path],
  file_metadata: Dict[Path, Dict],
  stream_id: str,
  ai_provider: str = "anthropic"
):
  """
    Background task to extract name images and optional AI names (no splitting).
  """
  log = logging.getLogger(__name__)
  log.info(f"Extracting names for {len(file_paths)} files in session {session_id}")

  try:
    session_repo = SessionRepository()
    session = session_repo.get_by_id(session_id)
    if not session:
      log.error(f"Session {session_id} not found")
      return

    session_data = session_repo.get_metadata(session_id) or {}
    page_transforms_by_hash = session_data.get("page_transforms", {})

    mock_roster = bool(session.metadata and session.metadata.get("mock_roster"))
    ai_name_extraction = True
    if session.metadata and "ai_name_extraction" in session.metadata:
      ai_name_extraction = bool(session.metadata.get("ai_name_extraction"))

    canvas_students = []
    if not mock_roster:
      canvas_interface = CanvasInterface(
        prod=session.use_prod_canvas,
        privacy_mode="none"
      )
      course = canvas_interface.get_course(session.course_id)
      assignment = course.get_assignment(session.assignment_id)
      students = assignment.get_students(include_names=True)
      canvas_students = [{"name": s.name, "user_id": s.user_id} for s in students]

    submission_repo = SubmissionRepository()
    existing_hashes = submission_repo.get_existing_hashes(session_id)

    # Filter out duplicate files
    new_file_paths = []
    for file_path in file_paths:
      file_hash = file_metadata[file_path]["hash"]
      if file_hash in existing_hashes:
        log.info(
          f"Skipping duplicate file: {file_path.name} (hash={file_hash[:8]}..., already processed)"
        )
      else:
        new_file_paths.append(file_path)

    if not new_file_paths:
      log.info("No new files to process (all were duplicates)")
      session_repo.update_status(
        session_id,
        SessionStatus.NAME_MATCHING_NEEDED if not mock_roster else SessionStatus.AWAITING_ALIGNMENT,
        "All uploaded files were duplicates - no new exams added"
      )
      return

    file_paths = new_file_paths
    start_document_id = submission_repo.get_max_document_id(session_id) + 1
    base_total = session.total_exams
    base_processed = session.processed_exams
    base_matched = session.matched_exams

    main_loop = asyncio.get_event_loop()
    total_steps = len(file_paths)
    current_step = {"count": 0}

    def update_progress(processed, matched, message):
      total = base_total + len(file_paths)
      processed_count = base_processed + processed
      matched_count = base_matched + matched

      current_step["count"] += 1
      progress_repo = SessionRepository()
      progress_session = progress_repo.get_by_id(session_id)
      progress_session.total_exams = total
      progress_session.processed_exams = processed_count
      progress_session.matched_exams = matched_count
      progress_session.processing_message = message
      progress_repo.update(progress_session)

      progress_percent = min(100,
                             int((current_step["count"] / total_steps) * 100))

      try:
        asyncio.run_coroutine_threadsafe(
          sse.send_event(
            stream_id, "progress", {
              "total": total,
              "processed": processed_count,
              "matched": matched_count,
              "progress": progress_percent,
              "current_step": current_step["count"],
              "total_steps": total_steps,
              "message": message
            }), main_loop)
      except Exception as e:
        log.error(f"Failed to send SSE event: {e}")

    processor = ExamProcessor(ai_provider=ai_provider)
    submissions_to_create = []
    matched_count = 0
    unmatched_students = canvas_students.copy()

    for index, pdf_path in enumerate(file_paths):
      document_id = index + start_document_id
      log.info(
        f"Extracting name for exam {index + 1}/{len(file_paths)}: {pdf_path.name}"
      )
      update_progress(index, matched_count,
                      f"Extracting name {index + 1}/{len(file_paths)}: {pdf_path.name}")

      if mock_roster:
        name_image = processor.extract_name_image(pdf_path)
        student_index = base_total + index + 1
        student_name = f"Student {student_index}"
        approximate_name = student_name
        canvas_user_id = -(student_index)
        matched_count += 1
      else:
        if ai_name_extraction:
          approximate_name, name_image = processor.extract_name(
            pdf_path,
            student_names=[s["name"] for s in unmatched_students]
          )
        else:
          name_image = processor.extract_name_image(pdf_path)
          approximate_name = ""
        suggested_match = None
        match_confidence = 0
        if approximate_name:
          suggested_match, match_confidence = processor._find_suggested_match(
            approximate_name,
            unmatched_students
          )

        if suggested_match and match_confidence >= NAME_SIMILARITY_THRESHOLD:
          student_name = suggested_match["name"]
          canvas_user_id = suggested_match["user_id"]
          unmatched_students = [
            student for student in unmatched_students
            if student["user_id"] != suggested_match["user_id"]
          ]
          matched_count += 1
          log.info(
            "Auto-accepted name match for %s: %s (%s%%)",
            pdf_path.name,
            student_name,
            match_confidence
          )
        else:
          student_name = None
          canvas_user_id = None

      submission = Submission(
        id=0,
        session_id=session_id,
        document_id=document_id,
        approximate_name=approximate_name,
        name_image_data=name_image,
        student_name=student_name,
        display_name=None,
        canvas_user_id=canvas_user_id,
        page_mappings=[],
        total_score=None,
        graded_at=None,
        file_hash=file_metadata[pdf_path]["hash"],
        original_filename=file_metadata[pdf_path]["original_filename"],
        exam_pdf_data=None
      )
      submissions_to_create.append(submission)

    if submissions_to_create:
      submission_repo.bulk_create(submissions_to_create)

    next_status = SessionStatus.NAME_MATCHING_NEEDED
    if mock_roster:
      next_status = SessionStatus.AWAITING_ALIGNMENT
    session_repo.update_status(session_id, next_status, "Name extraction complete")

    await sse.send_event(
      stream_id, "complete", {
        "total": len(file_paths),
        "matched": matched_count,
        "unmatched": len(file_paths) - matched_count,
        "message": f"Name extraction complete: {len(file_paths)} exams processed"
      })

  except Exception as e:
    log.error(f"Error extracting names: {e}", exc_info=True)
    await sse.send_event(stream_id, "error", {
      "error": str(e),
      "message": f"Name extraction failed: {str(e)}"
    })
    error_repo = SessionRepository()
    error_repo.update_status(session_id, SessionStatus.ERROR, f"Name extraction failed: {str(e)}")


async def process_exam_splits(
  session_id: int,
  file_paths: List[Path],
  file_metadata: Dict[Path, Dict],
  stream_id: str,
  manual_split_points: Optional[Dict[int, List[int]]] = None,
  skip_first_region: bool = True,
  last_page_blank: bool = False,
  ai_provider: str = "anthropic"
):
  """
    Background task to split exams into problems after name matching.
  """
  log = logging.getLogger(__name__)
  log.info(f"Splitting {len(file_paths)} files for session {session_id}")

  try:
    session_repo = SessionRepository()
    session = session_repo.get_by_id(session_id)
    if not session:
      log.error(f"Session {session_id} not found")
      return

    session_data = session_repo.get_metadata(session_id) or {}
    page_transforms_by_hash = session_data.get("page_transforms", {})

    submission_repo = SubmissionRepository()
    problem_repo = ProblemRepository()
    processed_submission_ids = problem_repo.get_submission_ids_with_problems(session_id)

    file_paths_to_process = []
    for file_path in file_paths:
      file_hash = file_metadata[file_path]["hash"]
      submission = submission_repo.get_by_file_hash(session_id, file_hash)
      if not submission:
        log.warning(f"No submission found for file {file_path.name}; skipping")
        continue
      if submission.id in processed_submission_ids:
        log.info(f"Skipping already-processed submission {submission.id}")
        continue
      file_paths_to_process.append(file_path)

    if not file_paths_to_process:
      session_repo.update_status(session_id, SessionStatus.READY, "No new exams to split")
      return

    def estimate_regions_per_exam(sample_path: Path) -> int:
      try:
        import fitz
        doc = fitz.open(str(sample_path))
        total_pages = doc.page_count
        doc.close()
      except Exception:
        return 1

      linear_splits = []
      for page_num in range(total_pages):
        page_height = 1.0
        if manual_split_points and page_num in manual_split_points:
          for pct in manual_split_points.get(page_num, []):
            y_pos = pct * page_height
            if abs(y_pos - page_height) < 0.001 and page_num < total_pages - 1:
              linear_splits.append((page_num + 1, 0.0))
            else:
              linear_splits.append((page_num, y_pos))

      linear_splits.sort(key=lambda x: (x[0], x[1]))
      unique_splits = []
      for split in linear_splits:
        if not unique_splits or split != unique_splits[-1]:
          unique_splits.append(split)
      linear_splits = unique_splits

      if not linear_splits:
        linear_splits = [(0, 0.0), (total_pages - 1, 1.0)]

      if linear_splits[0] != (0, 0.0):
        linear_splits.insert(0, (0, 0.0))
      if linear_splits[-1] != (total_pages - 1, 1.0):
        linear_splits.append((total_pages - 1, 1.0))

      if last_page_blank and total_pages > 0:
        last_page_num = total_pages - 1
        linear_splits = [(page, y) for page, y in linear_splits
                         if page < last_page_num]
        if total_pages > 1 and linear_splits:
          expected_end = (last_page_num - 1, 1.0)
          if linear_splits[-1] != expected_end:
            linear_splits.append(expected_end)

      regions = max(0, len(linear_splits) - 1)
      if skip_first_region and regions > 0:
        regions -= 1

      return max(1, regions)

    base_total = session.total_exams
    base_processed = session.processed_exams
    base_matched = session.matched_exams

    main_loop = asyncio.get_event_loop()
    regions_per_exam = estimate_regions_per_exam(file_paths_to_process[0])
    total_steps = len(file_paths_to_process) * (1 + regions_per_exam * len(PRESCAN_DPI_STEPS))
    current_step = {"count": 0}

    def update_progress(processed, matched, message, step_increment: int = 1):
      total = base_total
      processed_count = base_processed + processed
      matched_count = base_matched + matched

      current_step["count"] += step_increment
      progress_repo = SessionRepository()
      progress_session = progress_repo.get_by_id(session_id)
      progress_session.total_exams = total
      progress_session.processed_exams = processed_count
      progress_session.matched_exams = matched_count
      progress_session.processing_message = message
      progress_repo.update(progress_session)

      progress_percent = min(100,
                             int((current_step["count"] / total_steps) * 100))

      try:
        asyncio.run_coroutine_threadsafe(
          sse.send_event(
            stream_id, "progress", {
              "total": total,
              "processed": processed_count,
              "matched": matched_count,
              "progress": progress_percent,
              "current_step": current_step["count"],
              "total_steps": total_steps,
              "message": message
            }), main_loop)
      except Exception as e:
        log.error(f"Failed to send SSE event: {e}")

    processor = ExamProcessor(ai_provider=ai_provider)
    loop = asyncio.get_event_loop()
    matched, unmatched = await loop.run_in_executor(
      None,
      lambda: processor.process_exams(
        input_files=file_paths_to_process,
        canvas_students=[],
        progress_callback=update_progress,
        document_id_offset=0,
        file_metadata=file_metadata,
        manual_split_points=manual_split_points,
        skip_first_region=skip_first_region,
        last_page_blank=last_page_blank,
        skip_name_extraction=True,
        mock_roster=False
      ))

    metadata_repo = ProblemMetadataRepository()
    problem_max_points = metadata_repo.get_all_max_points(session_id)

    with with_transaction() as repos:
      all_submissions_data = matched + unmatched
      all_problems = []
      max_points_to_upsert = {}

      for sub_dto in all_submissions_data:
        if not sub_dto.file_hash:
          continue
        submission = repos.submissions.get_by_file_hash(session_id, sub_dto.file_hash)
        if not submission:
          continue
        if submission.id in processed_submission_ids:
          continue

        repos.submissions.update_processing_data(
          submission.id,
          sub_dto.page_mappings,
          sub_dto.pdf_data
        )

        for prob_dto in sub_dto.problems:
          problem_number = prob_dto.problem_number
          existing_max = repos.metadata.get_max_points(session_id, problem_number)
          if existing_max is not None:
            max_points = existing_max
          else:
            max_points = prob_dto.max_points
            if max_points is not None:
              max_points_to_upsert[problem_number] = max_points

          region_coords = prob_dto.region_coords
          transforms_for_file = page_transforms_by_hash.get(sub_dto.file_hash, {})
          if transforms_for_file and region_coords:
            page_transforms = {}
            start_page = region_coords.get("page_number")
            if start_page is not None:
              transform = transforms_for_file.get(str(start_page))
              if transform:
                page_transforms[str(start_page)] = transform
            end_page = region_coords.get("end_page_number")
            if end_page is not None and end_page != start_page:
              transform = transforms_for_file.get(str(end_page))
              if transform:
                page_transforms[str(end_page)] = transform
            if page_transforms:
              region_coords = dict(region_coords)
              region_coords["page_transforms"] = page_transforms

          problem = Problem(
            id=0,
            session_id=session_id,
            submission_id=submission.id,
            problem_number=problem_number,
            graded=False,
            is_blank=prob_dto.is_blank,
            blank_confidence=prob_dto.blank_confidence,
            blank_method=prob_dto.blank_method,
            blank_reasoning=prob_dto.blank_reasoning,
            max_points=max_points,
            region_coords=region_coords,
            qr_encrypted_data=prob_dto.qr_encrypted_data
          )
          all_problems.append(problem)

      if all_problems:
        repos.problems.bulk_create(all_problems)
      for problem_num, max_pts in max_points_to_upsert.items():
        repos.metadata.upsert_max_points(session_id, problem_num, max_pts)

      repos.sessions.update_status(session_id, SessionStatus.READY)

    await sse.send_event(
      stream_id, "complete", {
        "total": len(file_paths_to_process),
        "matched": len(file_paths_to_process),
        "unmatched": 0,
        "message": "Splitting complete"
      })

  except Exception as e:
    log.error(f"Error splitting exams: {e}", exc_info=True)
    await sse.send_event(stream_id, "error", {
      "error": str(e),
      "message": f"Splitting failed: {str(e)}"
    })
    error_repo = SessionRepository()
    error_repo.update_status(session_id, SessionStatus.ERROR, f"Splitting failed: {str(e)}")
