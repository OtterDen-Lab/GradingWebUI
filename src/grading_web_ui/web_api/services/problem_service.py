"""
Service for problem-specific business logic.
Handles operations on individual problems like image extraction.
"""
import base64
import io
import json
import logging
import math
from pathlib import Path
from typing import Tuple, Optional, Dict
import fitz  # PyMuPDF
from PIL import Image

from ..repositories import ProblemRepository, SubmissionRepository

log = logging.getLogger(__name__)


class ProblemService:
  """Business logic for problem operations"""

  def __init__(self):
    self.problem_repo = ProblemRepository()
    self.submission_repo = SubmissionRepository()

  def get_problem_image(self, problem_id: int, dpi: int = 150) -> str:
    """
    Extract and return the image for a specific problem.

    Args:
        problem_id: The problem ID
        dpi: Resolution for image extraction (default 150)

    Returns:
        Base64 encoded PNG image
    """
    problem = self.problem_repo.get_by_id(problem_id)
    if not problem:
      raise ValueError(f"Problem {problem_id} not found")

    submission = self.submission_repo.get_by_id(problem.submission_id)
    if not submission:
      raise ValueError(
        f"Submission {problem.submission_id} not found for problem {problem_id}"
      )

    if not submission.exam_pdf_data:
      raise ValueError(
        f"No PDF data for submission {submission.id} (problem {problem_id})")

    # Parse region coordinates
    region_coords = json.loads(problem.region_coords)

    # Extract image using coordinates
    return self.extract_image_from_pdf_data(
      pdf_base64=submission.exam_pdf_data,
      page_number=region_coords["page_number"],
      region_y_start=region_coords["region_y_start"],
      region_y_end=region_coords["region_y_end"],
      end_page_number=region_coords.get("end_page_number"),
      end_region_y=region_coords.get("end_region_y"),
      region_y_start_pct=region_coords.get("region_y_start_pct"),
      region_y_end_pct=region_coords.get("region_y_end_pct"),
      end_region_y_pct=region_coords.get("end_region_y_pct"),
      page_transforms=region_coords.get("page_transforms"),
      dpi=dpi)

  def extract_image_from_pdf_data(
      self,
      pdf_base64: str,
      page_number: int,
      region_y_start: float,
      region_y_end: float,
      end_page_number: Optional[int] = None,
      end_region_y: Optional[float] = None,
      region_y_start_pct: Optional[float] = None,
      region_y_end_pct: Optional[float] = None,
      end_region_y_pct: Optional[float] = None,
      page_transforms: Optional[Dict] = None,
      dpi: int = 150) -> str:
    """
    Extract a region from PDF data as an image.
    Supports both single-page and cross-page regions.

    Args:
        pdf_base64: Base64 encoded PDF data
        page_number: Starting page number (0-indexed)
        region_y_start: Y coordinate of region start
        region_y_end: Y coordinate of region end (on start page if single-page)
        end_page_number: Optional end page number for cross-page regions
        end_region_y: Optional end Y coordinate for cross-page regions
        dpi: Resolution for image extraction (default 150)

    Returns:
        Base64 encoded PNG image
    """
    # Decode PDF from base64
    pdf_bytes = base64.b64decode(pdf_base64)
    pdf_document = fitz.open("pdf", pdf_bytes)

    try:
      # Determine actual end page and Y
      actual_end_page = end_page_number if end_page_number is not None else page_number
      actual_end_y = end_region_y if end_region_y is not None else region_y_end

      # Extract using the opened document
      image_base64, _ = self.extract_image_from_document(
        pdf_document=pdf_document,
        start_page=page_number,
        start_y=region_y_start,
        end_page=actual_end_page,
        end_y=actual_end_y,
        start_y_pct=region_y_start_pct,
        end_y_pct=region_y_end_pct,
        end_page_y_pct=end_region_y_pct,
        page_transforms=page_transforms,
        dpi=dpi)

      return image_base64
    finally:
      pdf_document.close()

  def extract_image_from_document(
      self,
      pdf_document: fitz.Document,
      start_page: int,
      start_y: float,
      end_page: int,
      end_y: float,
      start_y_pct: Optional[float] = None,
      end_y_pct: Optional[float] = None,
      end_page_y_pct: Optional[float] = None,
      page_transforms: Optional[Dict] = None,
      dpi: int = 150) -> Tuple[str, int]:
    """
    Extract a region from an already-opened PDF document.
    Supports both single-page and cross-page regions.

    This is the canonical implementation of image extraction.

    Args:
        pdf_document: Opened PyMuPDF document
        start_page: Starting page number (0-indexed)
        start_y: Starting Y coordinate
        end_page: Ending page number (0-indexed)
        end_y: Ending Y coordinate
        dpi: Resolution for image extraction (default 150)

    Returns:
        Tuple of (base64_image, total_height_pixels)
    """
    if start_page == end_page:
      # Single page region
      return self._extract_single_page_region(pdf_document, start_page,
                                               start_y, end_y,
                                               start_y_pct, end_y_pct,
                                               page_transforms, dpi)
    else:
      # Cross-page region
      return self._extract_cross_page_region(pdf_document, start_page, start_y,
                                              end_page, end_y,
                                              start_y_pct, end_page_y_pct,
                                              page_transforms, dpi)

  def _extract_single_page_region(self, pdf_document: fitz.Document,
                                   page_number: int, start_y: float,
                                   end_y: float,
                                   start_y_pct: Optional[float],
                                   end_y_pct: Optional[float],
                                   page_transforms: Optional[Dict],
                                   dpi: int) -> Tuple[str, int]:
    """Extract a region from a single page"""
    image = self._render_transformed_region(
      pdf_document,
      page_number=page_number,
      region_start=start_y,
      region_end=end_y,
      region_start_pct=start_y_pct,
      region_end_pct=end_y_pct,
      page_transforms=page_transforms,
      dpi=dpi
    )
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_base64, image.height

  def _extract_cross_page_region(self, pdf_document: fitz.Document,
                                  start_page: int, start_y: float,
                                  end_page: int, end_y: float,
                                  start_y_pct: Optional[float],
                                  end_page_y_pct: Optional[float],
                                  page_transforms: Optional[Dict],
                                  dpi: int) -> Tuple[str, int]:
    """Extract a region that spans multiple pages and merge vertically"""
    log.info(
      f"Extracting cross-page region from page {start_page} (y={start_y}) to page {end_page} (y={end_y})"
    )

    page_images = []
    total_height = 0

    use_pct = start_y_pct is not None or end_page_y_pct is not None

    # Extract first page (from start_y to bottom)
    first_page = pdf_document[start_page]
    if start_y < first_page.rect.height:
      img = self._render_transformed_region(
        pdf_document,
        page_number=start_page,
        region_start=start_y,
        region_end=first_page.rect.height,
        region_start_pct=start_y_pct if use_pct else None,
        region_end_pct=1.0 if use_pct else None,
        page_transforms=page_transforms,
        dpi=dpi
      )
      page_images.append(img)
      total_height += img.height

    # Extract middle pages (full pages)
    for page_num in range(start_page + 1, end_page):
      middle_page = pdf_document[page_num]
      img = self._render_transformed_region(
        pdf_document,
        page_number=page_num,
        region_start=0,
        region_end=middle_page.rect.height,
        region_start_pct=0.0 if use_pct else None,
        region_end_pct=1.0 if use_pct else None,
        page_transforms=page_transforms,
        dpi=dpi
      )
      page_images.append(img)
      total_height += img.height

    # Extract last page (from top to end_y)
    last_page = pdf_document[end_page]
    if end_y > 0:
      img = self._render_transformed_region(
        pdf_document,
        page_number=end_page,
        region_start=0,
        region_end=end_y,
        region_start_pct=0.0 if use_pct else None,
        region_end_pct=end_page_y_pct if use_pct else None,
        page_transforms=page_transforms,
        dpi=dpi
      )
      page_images.append(img)
      total_height += img.height

    # Merge all page images vertically
    if not page_images:
      # No valid regions found - create minimal image
      img = Image.new('RGB', (int(first_page.rect.width), 1), color='white')
      buffer = io.BytesIO()
      img.save(buffer, format='PNG')
      return base64.b64encode(buffer.getvalue()).decode('utf-8'), 1

    # Get width from first image (all should be same width)
    merged_width = page_images[0].width

    # Create merged image
    merged_image = Image.new('RGB', (merged_width, total_height))

    # Paste each page image
    current_y = 0
    for img in page_images:
      merged_image.paste(img, (0, current_y))
      current_y += img.height

    # Convert to base64
    buffer = io.BytesIO()
    merged_image.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return img_base64, total_height

  def _render_transformed_region(
      self,
      pdf_document: fitz.Document,
      page_number: int,
      region_start: float,
      region_end: float,
      region_start_pct: Optional[float],
      region_end_pct: Optional[float],
      page_transforms: Optional[Dict],
      dpi: int) -> Image.Image:
    page = pdf_document[page_number]
    pix = page.get_pixmap(dpi=dpi)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    if image.mode != 'RGB':
      image = image.convert('RGB')

    transform = None
    if page_transforms:
      transform = page_transforms.get(str(page_number))

    target_width = transform.get("target_width") if transform else None
    target_height = transform.get("target_height") if transform else None
    if target_width and target_height and image.size != (target_width, target_height):
      image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    base_width, base_height = image.size
    rotation_deg = transform.get("rotation_deg", 0.0) if transform else 0.0
    if rotation_deg:
      image = image.rotate(rotation_deg, expand=True, fillcolor='white')

    rotated_width, rotated_height = image.size
    pad_left = transform.get("pad_left", 0) if transform else 0
    pad_top = transform.get("pad_top", 0) if transform else 0
    padded_width = transform.get("padded_width", rotated_width) if transform else rotated_width
    padded_height = transform.get("padded_height", rotated_height) if transform else rotated_height

    if padded_width != rotated_width or padded_height != rotated_height:
      padded = Image.new('RGB', (padded_width, padded_height), color='white')
      padded.paste(image, (pad_left, pad_top))
      image = padded

    offset_x = transform.get("offset_x", 0) if transform else 0
    offset_y = transform.get("offset_y", 0) if transform else 0
    canvas_width = transform.get("canvas_width", image.size[0]) if transform else image.size[0]
    canvas_height = transform.get("canvas_height", image.size[1]) if transform else image.size[1]

    if canvas_width != image.size[0] or canvas_height != image.size[1] or offset_x or offset_y:
      canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
      canvas.paste(image, (offset_x, offset_y))
      image = canvas

    use_pct = region_start_pct is not None and region_end_pct is not None

    def transform_point(x: float, y: float) -> Tuple[float, float]:
      cx, cy = base_width / 2.0, base_height / 2.0
      if rotation_deg:
        theta = math.radians(rotation_deg)
        dx = x - cx
        dy = y - cy
        x = (dx * math.cos(theta) - dy * math.sin(theta)) + cx
        y = (dx * math.sin(theta) + dy * math.cos(theta)) + cy
        cx2, cy2 = rotated_width / 2.0, rotated_height / 2.0
        x += (cx2 - cx)
        y += (cy2 - cy)

      x += pad_left + offset_x
      y += pad_top + offset_y
      return x, y

    if use_pct:
      canvas_height = image.size[1]
      top = max(0, int(round(region_start_pct * canvas_height)))
      bottom = min(image.size[1], int(round(region_end_pct * canvas_height)))
      left = 0
      right = image.size[0]
    else:
      y_start_px = (region_start / page.rect.height) * base_height
      y_end_px = (region_end / page.rect.height) * base_height
      points = [
        transform_point(0, y_start_px),
        transform_point(base_width, y_start_px),
        transform_point(0, y_end_px),
        transform_point(base_width, y_end_px)
      ]
      xs = [p[0] for p in points]
      ys = [p[1] for p in points]

      left = max(0, int(min(xs)))
      right = min(image.size[0], int(max(xs)))
      top = max(0, int(min(ys)))
      bottom = min(image.size[1], int(max(ys)))

    anchor_aligned_y = None
    anchor_aligned_x = None
    ref_anchor_y = None
    ref_anchor_x = None
    if transform:
      anchor_aligned_y = transform.get("anchor_aligned_y")
      anchor_aligned_x = transform.get("anchor_aligned_x")
      ref_anchor_y = transform.get("ref_anchor_y")
      ref_anchor_x = transform.get("ref_anchor_x")

    log.debug(
      "Render page %s: rotate=%.2f pad=(%s,%s) offset=(%s,%s) canvas=%sx%s target=%sx%s crop=(%s,%s,%s,%s) pct=%s pct_range=(%s,%s) anchor_aligned=(%s,%s) ref_anchor=(%s,%s)",
      page_number + 1,
      rotation_deg,
      pad_left,
      pad_top,
      offset_x,
      offset_y,
      canvas_width,
      canvas_height,
      target_width or base_width,
      target_height or base_height,
      left,
      top,
      right,
      bottom,
      use_pct,
      region_start_pct,
      region_end_pct,
      anchor_aligned_y,
      anchor_aligned_x,
      ref_anchor_y,
      ref_anchor_x
    )

    if right <= left or bottom <= top:
      return Image.new('RGB', (1, 1), color='white')

    return image.crop((left, top, right, bottom))
