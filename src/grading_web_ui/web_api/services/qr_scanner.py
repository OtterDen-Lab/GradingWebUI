"""
QR code scanning service for quiz questions.

This service scans QR codes from exam problem images and extracts:
- Question number
- Maximum points
- Encrypted question metadata (passed to QuizGeneration for decryption)
"""
import base64
import json
import logging
from typing import Optional, Dict, List
from pathlib import Path
from PIL import Image
import io

log = logging.getLogger(__name__)

# Try to import pyzbar for QR code scanning
try:
  from pyzbar import pyzbar
  PYZBAR_AVAILABLE = True
except Exception as exc:
  log.warning(
    "pyzbar/zbar not available - QR code scanning will not be available (%s)",
    exc
  )
  PYZBAR_AVAILABLE = False


class QRScanner:
  """Service for scanning and processing QR codes from exam problems."""

  def __init__(self):
    """Initialize QR scanner."""
    self.available = PYZBAR_AVAILABLE
    if not PYZBAR_AVAILABLE:
      log.warning("QR scanner unavailable: pyzbar/zbar not available")

  def _decode_qr_codes(self, image: Image.Image) -> List[Dict]:
    """Decode QR codes from a PIL image into raw records."""
    qr_codes = pyzbar.decode(image)
    if not qr_codes:
      return []

    results = []
    for qr in qr_codes:
      try:
        qr_data = qr.data.decode('utf-8')
        qr_json = json.loads(qr_data)
      except Exception:
        continue

      question_number = qr_json.get('q')
      max_points = qr_json.get('pts')
      if question_number is None or max_points is None:
        continue

      rect = qr.rect
      results.append({
        "question_number": int(question_number),
        "max_points": float(max_points),
        "encrypted_data": qr_json.get('s'),
        "rect": rect
      })

    return results

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
      dpi_steps = [150, 300, 600, 900]

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
                                 dpi_steps: Optional[List[int]] = None) -> Dict[int, List[Dict]]:
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
      dpi_steps = [150, 300, 600, 900]

    try:
      pdf_document = fitz.open(str(pdf_path))
      for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        if image.mode != 'RGB':
          image = image.convert('RGB')

        scan_result = self._scan_image_step_up(
          image,
          dpi_steps=dpi_steps,
          debug_label=f"{pdf_path.name} page {page_number}"
        )

        if not scan_result:
          continue

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
            "height": rect.height * scale
          })

        if page_results:
          results[page_number] = page_results
    except Exception as e:
      log.error(f"Error scanning QR positions from PDF: {e}", exc_info=True)
    finally:
      try:
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
