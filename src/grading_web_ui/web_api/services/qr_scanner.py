"""
QR code scanning service for quiz questions.

This service scans QR codes from exam problem images and extracts:
- Question number
- Maximum points
- Encrypted question metadata (passed to QuizGeneration for decryption)
"""
import base64
import concurrent.futures
import json
import logging
import math
import os
import platform
import threading
from typing import Optional, Dict, List
from pathlib import Path
from PIL import Image
import io
from dataclasses import dataclass

log = logging.getLogger(__name__)


def _configure_macos_zbar_library_path() -> bool:
  """
  Add common Homebrew zbar library locations to DYLD fallback path.

  Returns:
      True if DYLD_FALLBACK_LIBRARY_PATH was updated, False otherwise.
  """
  if platform.system() != "Darwin":
    return False

  candidates = []
  brew_prefix = os.getenv("HOMEBREW_PREFIX")
  if brew_prefix:
    candidates.append(Path(brew_prefix) / "opt" / "zbar" / "lib")

  candidates.extend([
    Path("/opt/homebrew/opt/zbar/lib"),
    Path("/usr/local/opt/zbar/lib")
  ])

  existing_paths = []
  for candidate in candidates:
    if candidate.exists():
      path_str = str(candidate)
      if path_str not in existing_paths:
        existing_paths.append(path_str)

  if not existing_paths:
    return False

  current = os.getenv("DYLD_FALLBACK_LIBRARY_PATH", "")
  current_paths = [p for p in current.split(":") if p]
  new_paths = [p for p in existing_paths if p not in current_paths] + current_paths

  if new_paths == current_paths:
    return False

  os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(new_paths)
  return True


def _load_pyzbar():
  """Load pyzbar with a macOS/Homebrew zbar fallback."""
  try:
    from pyzbar import pyzbar as module
    return module, True, None
  except Exception as exc:
    first_error = exc

  if _configure_macos_zbar_library_path():
    try:
      from pyzbar import pyzbar as module
      return module, True, None
    except Exception as exc:
      return None, False, exc

  return None, False, first_error


pyzbar, PYZBAR_AVAILABLE, _PYZBAR_IMPORT_ERROR = _load_pyzbar()
if not PYZBAR_AVAILABLE:
  log.warning(
    "pyzbar/zbar not available - QR code scanning will not be available (%s)",
    _PYZBAR_IMPORT_ERROR
  )


_PYZBAR_DECODE_LOCK = threading.Lock()


@dataclass
class _QRRect:
  left: float
  top: float
  width: float
  height: float


class QRScanner:
  """Service for scanning and processing QR codes from exam problems."""

  _unavailable_warning_emitted = False

  def __init__(self):
    """Initialize QR scanner."""
    self.available = PYZBAR_AVAILABLE
    if (not PYZBAR_AVAILABLE and
        not QRScanner._unavailable_warning_emitted):
      log.warning("QR scanner unavailable: pyzbar/zbar not available")
      QRScanner._unavailable_warning_emitted = True

  @staticmethod
  def _estimate_angle(polygon_points) -> float:
    if not polygon_points or len(polygon_points) < 2:
      return 0.0
    points = [(getattr(p, "x", p[0]), getattr(p, "y", p[1]))
              for p in polygon_points]
    points_sorted = sorted(points, key=lambda pt: (pt[1], pt[0]))
    p1, p2 = points_sorted[0], points_sorted[1]
    if p2[0] < p1[0]:
      p1, p2 = p2, p1
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

  @staticmethod
  def _payload_from_text(payload_text: str) -> Optional[Dict]:
    try:
      qr_json = json.loads(payload_text)
    except Exception:
      return None

    if not isinstance(qr_json, dict):
      return None

    question_number = qr_json.get('q')
    max_points = qr_json.get('pts')
    if max_points is None:
      # Backward/forward compatibility across QR schema versions
      max_points = qr_json.get('p')
    if max_points is None:
      max_points = qr_json.get('points')
    if max_points is None:
      max_points = qr_json.get('max_points')
    if question_number is None or max_points is None:
      return None

    return {
      "question_number": int(question_number),
      "max_points": float(max_points),
      "encrypted_data": qr_json.get('s')
    }

  def _decode_qr_codes_pyzbar(self, image: Image.Image) -> List[Dict]:
    if not PYZBAR_AVAILABLE:
      return []

    try:
      with _PYZBAR_DECODE_LOCK:
        symbols = getattr(getattr(pyzbar, "ZBarSymbol", None), "QRCODE", None)
        if symbols is not None:
          qr_codes = pyzbar.decode(image, symbols=[symbols])
        else:
          qr_codes = pyzbar.decode(image)
    except Exception as exc:
      log.debug("pyzbar decode failed: %s", exc)
      return []

    results = []
    for qr in qr_codes:
      payload = self._payload_from_text(qr.data.decode('utf-8'))
      if not payload:
        continue
      payload.update({
        "rect": qr.rect,
        "angle": self._estimate_angle(qr.polygon) if hasattr(qr, "polygon") else 0.0
      })
      results.append(payload)

    return results

  @staticmethod
  def _points_to_rect(points) -> _QRRect:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    left = min(xs)
    top = min(ys)
    return _QRRect(
      left=left,
      top=top,
      width=max(xs) - left,
      height=max(ys) - top
    )

  def _decode_qr_codes_opencv(self, image: Image.Image) -> List[Dict]:
    try:
      import cv2
      import numpy as np
    except Exception:
      return []

    image_rgb = image.convert('RGB')
    image_array = np.array(image_rgb)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    detector = cv2.QRCodeDetector()
    results = []

    try:
      ok, decoded_info, points, _ = detector.detectAndDecodeMulti(image_bgr)
      if ok and decoded_info is not None:
        for idx, text in enumerate(decoded_info):
          if not text:
            continue
          payload = self._payload_from_text(text)
          if not payload:
            continue

          points_for_code = []
          if points is not None and len(points) > idx:
            points_for_code = [
              (float(p[0]), float(p[1])) for p in points[idx]
            ]

          rect = self._points_to_rect(points_for_code) if points_for_code else _QRRect(
            left=0,
            top=0,
            width=float(image_rgb.width),
            height=float(image_rgb.height)
          )
          payload.update({
            "rect": rect,
            "angle": self._estimate_angle(points_for_code)
          })
          results.append(payload)
    except Exception as exc:
      log.debug("OpenCV detectAndDecodeMulti failed: %s", exc)

    if results:
      return results

    try:
      text, points, _ = detector.detectAndDecode(image_bgr)
      if not text:
        return []

      payload = self._payload_from_text(text)
      if not payload:
        return []

      points_for_code = []
      if points is not None:
        try:
          points_for_code = [
            (float(p[0]), float(p[1])) for p in points.reshape(-1, 2)
          ]
        except Exception:
          points_for_code = []

      rect = self._points_to_rect(points_for_code) if points_for_code else _QRRect(
        left=0,
        top=0,
        width=float(image_rgb.width),
        height=float(image_rgb.height)
      )
      payload.update({
        "rect": rect,
        "angle": self._estimate_angle(points_for_code)
      })
      return [payload]
    except Exception as exc:
      log.debug("OpenCV detectAndDecode failed: %s", exc)
      return []

  def _decode_qr_codes(self, image: Image.Image) -> List[Dict]:
    """Decode QR codes from a PIL image into raw records."""
    results = self._decode_qr_codes_pyzbar(image)
    if results:
      return results

    return self._decode_qr_codes_opencv(image)

  def _scan_image_step_up(self, image: Image.Image,
                          dpi_steps: Optional[List[int]] = None,
                          debug_label: Optional[str] = None
                          ) -> Optional[Dict]:
    """
        Scan a single image using step-up DPI heuristics.

        Returns dict with keys:
          - dpi_used
          - scale_to_base
          - qr_codes (list of decoded QR dicts with rects)
    """
    if dpi_steps is None:
      dpi_steps = [150, 300, 600]

    for dpi in dpi_steps:
      scaled = image
      if dpi != 150:
        scale = dpi / 150.0
        new_size = (int(image.width * scale), int(image.height * scale))
        scaled = image.resize(new_size, Image.Resampling.LANCZOS)

      if scaled.mode != 'RGB':
        scaled = scaled.convert('RGB')

      decoded = self._decode_qr_codes(scaled)
      if decoded:
        if debug_label:
          log.debug(
            "QR scan: %s detected %s code(s) at %sdpi",
            debug_label,
            len(decoded),
            dpi
          )

        return {
          "dpi_used": dpi,
          "scale_to_base": 150.0 / dpi,
          "qr_codes": decoded
        }

    return None

  def _preprocess_image_for_qr(self, image: Image.Image) -> List[Image.Image]:
    """
        Apply multiple preprocessing strategies to improve QR detection.

        Args:
            image: PIL Image to preprocess

        Returns:
            List of preprocessed images to try scanning
        """
    from PIL import ImageEnhance, ImageFilter

    variants = []

    # 1. Original image
    variants.append(image.copy())

    # 2. Sharpened image (helps with blurry QR codes)
    sharpened = image.filter(ImageFilter.SHARPEN)
    variants.append(sharpened)

    # 3. High contrast (helps with faded QR codes)
    enhancer = ImageEnhance.Contrast(image)
    high_contrast = enhancer.enhance(2.0)
    variants.append(high_contrast)

    # 4. Grayscale with adaptive threshold simulation (convert to L mode)
    grayscale = image.convert('L')
    # Enhance contrast on grayscale
    enhancer_gray = ImageEnhance.Contrast(grayscale)
    high_contrast_gray = enhancer_gray.enhance(2.5)
    variants.append(high_contrast_gray.convert('RGB'))

    # 5. Sharpened + high contrast combo
    sharpened_contrast = sharpened.copy()
    enhancer_combo = ImageEnhance.Contrast(sharpened_contrast)
    combo = enhancer_combo.enhance(1.5)
    variants.append(combo)

    return variants

  def scan_qr_from_image(self, image_base64: str) -> Optional[Dict]:
    """
        Scan QR code from a base64-encoded image.
        Uses multiple preprocessing strategies to improve detection rate.

        Args:
            image_base64: Base64 encoded PNG/JPEG image

        Returns:
            Dict with QR code data if found, None otherwise.
            Format: {
                "question_number": int,
                "max_points": float,
                "question_type": str,
                "seed": int,
                "version": str
            }
        """
    if not self.available:
      log.debug("QR scanner not available, skipping scan")
      return None

    try:
      # Decode image
      image_bytes = base64.b64decode(image_base64)
      image = Image.open(io.BytesIO(image_bytes))

      # Convert to RGB if needed (pyzbar works best with RGB)
      if image.mode != 'RGB':
        image = image.convert('RGB')

      image_variants = self._preprocess_image_for_qr(image)

      for idx, variant in enumerate(image_variants):
        scan_result = self._scan_image_step_up(variant)
        if not scan_result:
          continue

        if idx > 0:
          log.info(f"QR code found using preprocessing strategy #{idx}")

        qr_codes = scan_result["qr_codes"]
        qr_code = qr_codes[0]
        question_number = qr_code["question_number"]
        max_points = qr_code["max_points"]
        encrypted_metadata = qr_code.get("encrypted_data")

        result = {
          "question_number": question_number,
          "max_points": float(max_points),
          "encrypted_data": encrypted_metadata
        }

        if encrypted_metadata:
          log.info(
            f"Successfully scanned QR code: Q{question_number}, {max_points} pts (has encrypted metadata) : \"{encrypted_metadata}\""
          )
        else:
          log.info(
            f"Successfully scanned QR code: Q{question_number}, {max_points} pts (no metadata)"
          )

        return result

      log.debug(
        "No QR codes found in image after trying all preprocessing strategies")
      return None

    except Exception as e:
      log.error(f"Error scanning QR code: {e}", exc_info=True)
      return None

  def scan_qr_from_region(self, pdf_base64: str, page_number: int,
                          region_y_start: int,
                          region_y_end: int) -> Optional[Dict]:
    """
        Extract a region from a PDF and scan for QR codes.

        Args:
            pdf_base64: Base64 encoded PDF
            page_number: 0-indexed page number
            region_y_start: Y coordinate of region start
            region_y_end: Y coordinate of region end

        Returns:
            Dict with QR code data if found, None otherwise
        """
    if not self.available:
      return None

    try:
      import fitz  # PyMuPDF

      # Decode PDF
      pdf_bytes = base64.b64decode(pdf_base64)
      pdf_document = fitz.open("pdf", pdf_bytes)

      # Get page
      page = pdf_document[page_number]

      # Create region rectangle
      region = fitz.Rect(0, region_y_start, page.rect.width, region_y_end)

      # Extract region as image
      problem_pdf = fitz.open()
      problem_page = problem_pdf.new_page(width=region.width,
                                          height=region.height)
      problem_page.show_pdf_page(problem_page.rect,
                                 pdf_document,
                                 page_number,
                                 clip=region)

      # Convert to PNG
      pix = problem_page.get_pixmap(dpi=150)
      img_bytes = pix.tobytes("png")
      img_base64 = base64.b64encode(img_bytes).decode("utf-8")

      # Cleanup
      problem_pdf.close()
      pdf_document.close()

      # Scan QR code from extracted image
      return self.scan_qr_from_image(img_base64)

    except Exception as e:
      log.error(f"Error scanning QR from PDF region: {e}", exc_info=True)
      return None

  def scan_qr_positions_from_pdf(self, pdf_path: Path,
                                 dpi_steps: Optional[List[int]] = None,
                                 progress_callback: Optional[callable] = None,
                                 max_workers: Optional[int] = None
                                 ) -> Dict[int, List[Dict]]:
    """
        Scan QR codes from each page in a PDF and return their positions.

        Args:
        pdf_path: Path to PDF file
        dpi_steps: List of DPIs to try in order (e.g., [150, 300, 600])

        Returns:
            Dict mapping page_number -> list of QR info dicts:
            {
              "question_number": int,
              "max_points": float,
              "x": int,
              "y": int,
              "width": int,
              "height": int
            }
    """
    if not self.available:
      return {}

    try:
      import fitz  # PyMuPDF
    except Exception as exc:
      log.warning("PyMuPDF not available for QR position scan: %s", exc)
      return {}

    results: Dict[int, List[Dict]] = {}

    if dpi_steps is None:
      dpi_steps = [150, 300, 600]

    pdf_document = None
    try:
      pdf_document = fitz.open(str(pdf_path))
      total_pages = pdf_document.page_count

      if total_pages == 0:
        pdf_document.close()
        return results

      if max_workers is None:
        max_workers = min(4, total_pages, os.cpu_count() or 1)
      max_workers = max(1, min(max_workers, total_pages))

      completed = {"count": 0}
      progress_lock = threading.Lock()

      def report_progress(page_number: int) -> None:
        if not progress_callback:
          return
        with progress_lock:
          completed["count"] += 1
          done = completed["count"]
        progress_callback(
          done,
          total_pages,
          f"Scanning QR codes ({done}/{total_pages}) in {pdf_path.name} (page {page_number + 1})"
        )

      def scan_page_from_doc(page) -> List[Dict]:
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        if image.mode != 'RGB':
          image = image.convert('RGB')

        scan_result = self._scan_image_step_up(
          image,
          dpi_steps=dpi_steps,
          debug_label=f"{pdf_path.name} page {page.number}"
        )
        if not scan_result:
          return []

        scale = scan_result["scale_to_base"]
        page_results: List[Dict] = []
        for qr_code in scan_result["qr_codes"]:
          rect = qr_code["rect"]
          page_results.append({
            "question_number": qr_code["question_number"],
            "max_points": qr_code["max_points"],
            "x": rect.left * scale,
            "y": rect.top * scale,
            "width": rect.width * scale,
            "height": rect.height * scale,
            "angle": qr_code.get("angle", 0.0)
          })
        return page_results

      def scan_page_in_thread(page_number: int) -> tuple[int, List[Dict]]:
        local_doc = fitz.open(str(pdf_path))
        try:
          page = local_doc[page_number]
          return page_number, scan_page_from_doc(page)
        finally:
          local_doc.close()

      if max_workers == 1:
        for page_number in range(total_pages):
          page = pdf_document[page_number]
          page_results = scan_page_from_doc(page)
          if page_results:
            results[page_number] = page_results
          report_progress(page_number)
      else:
        pdf_document.close()
        pdf_document = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
          futures = [
            executor.submit(scan_page_in_thread, page_number)
            for page_number in range(total_pages)
          ]
          for future in concurrent.futures.as_completed(futures):
            page_number, page_results = future.result()
            if page_results:
              results[page_number] = page_results
            report_progress(page_number)

      if pdf_document is not None:
        pdf_document.close()

    except Exception as e:
      log.error(f"Error scanning QR positions from PDF: {e}", exc_info=True)
      try:
        if pdf_document is not None:
          pdf_document.close()
      except Exception:
        pass

    return results

  def scan_multiple_regions(self, pdf_base64: str,
                            regions: List[Dict]) -> Dict[int, Optional[Dict]]:
    """
        Scan multiple regions for QR codes.

        Args:
            pdf_base64: Base64 encoded PDF
            regions: List of region dicts with keys:
                     - page_number
                     - region_y_start
                     - region_y_end
                     - problem_number

        Returns:
            Dict mapping problem_number -> QR data (or None if not found)
        """
    results = {}

    for region in regions:
      problem_number = region.get("problem_number")
      if not problem_number:
        continue

      qr_data = self.scan_qr_from_region(pdf_base64, region["page_number"],
                                         region["region_y_start"],
                                         region["region_y_end"])

      results[problem_number] = qr_data

    return results
