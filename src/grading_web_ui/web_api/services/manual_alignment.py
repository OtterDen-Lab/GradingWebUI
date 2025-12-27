"""
Manual alignment service - creates composite images and handles user-defined split points
"""
from typing import List, Dict, Optional, Tuple
import math
from pathlib import Path
import logging
import base64
import numpy as np
from PIL import Image
import io
import fitz

log = logging.getLogger(__name__)


class ManualAlignmentService:
  """Service for manual alignment of exam split points"""

  def suggest_split_points_from_composites(
      self,
      composites: Dict[int, str],
      qr_positions_by_file: Optional[Dict[Path, Dict[int, List[Dict]]]] = None,
      transforms_by_file: Optional[Dict[Path, Dict[int, Dict]]] = None
  ) -> Dict[int, List[int]]:
    """
    Suggest split points by detecting horizontal lines above QR codes.
    Returns page_number -> list of y positions (pixels).
    """
    if not composites:
      return {}

    suggested: Dict[int, List[int]] = {}
    min_qr_count = 3

    for page_num, image_base64 in composites.items():
      try:
        img_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
          img = img.convert('RGB')
      except Exception as exc:
        log.debug("Failed to decode composite page %s: %s", page_num + 1, exc)
        continue

      qr_candidates: Dict[Optional[int], List[Tuple[float, float, float, float]]] = {}
      if qr_positions_by_file and transforms_by_file:
        for pdf_path, page_positions in qr_positions_by_file.items():
          positions = page_positions.get(page_num, [])
          if not positions:
            continue
          transform = transforms_by_file.get(pdf_path, {}).get(page_num)
          if not transform:
            continue
          for pos in positions:
            x_pos = pos.get("x")
            y_pos = pos.get("y")
            width_pos = pos.get("width")
            height_pos = pos.get("height")
            if x_pos is None or y_pos is None:
              continue
            if width_pos is None or height_pos is None:
              continue
            question_number = pos.get("question_number")
            mapped = self._map_rect_to_composite(
              x_pos,
              y_pos,
              width_pos,
              height_pos,
              transform
            )
            if mapped:
              qr_candidates.setdefault(question_number, []).append(mapped)
      else:
        try:
          from .qr_scanner import QRScanner
          qr_scanner = QRScanner()
        except Exception:
          qr_scanner = None

        if qr_scanner and qr_scanner.available:
          scan_result = qr_scanner._scan_image_step_up(img)
          if scan_result and scan_result.get("qr_codes"):
            scale = scan_result.get("scale_to_base", 1.0)
            for code in scan_result["qr_codes"]:
              rect = code["rect"]
              question_number = code.get("question_number")
              qr_candidates.setdefault(question_number, []).append(
                (
                  rect.top * scale,
                  rect.left * scale,
                  (rect.top + rect.height) * scale,
                  (rect.left + rect.width) * scale
                )
              )

      if not qr_candidates:
        continue

      gray = np.array(img.convert('L'))
      height, width = gray.shape
      max_search = int(min(height * 0.25, 300))
      min_gap = int(max(5, height * 0.005))

      for question_number, rects in qr_candidates.items():
        if len(rects) < min_qr_count:
          continue
        line_candidates: List[float] = []
        for qr_top, qr_left, qr_bottom, qr_right in rects:
          qr_height = max(1.0, qr_bottom - qr_top)
          qr_width = max(1.0, qr_right - qr_left)
          x_start = max(0, int(qr_left - (2.0 * qr_width)))
          x_end = min(width, int(qr_right + (0.2 * qr_width)))
          y_start = int(max(0, qr_top - (2.0 * qr_height)))
          y_end = int(max(0, qr_top - (0.15 * qr_height)))
          line_y = self._find_horizontal_line_above(
            gray,
            qr_top,
            max_search=max_search,
            min_gap=min_gap,
            x_start=x_start,
            x_end=x_end,
            y_start=y_start,
            y_end=y_end
          )
          if line_y is not None:
            line_candidates.append(line_y)
        if not line_candidates:
          continue
        line_y = max(line_candidates)
        if page_num != 0 and line_y < int(height * 0.08):
          split_y = 0
        else:
          split_y = int(round(line_y))
        suggested.setdefault(page_num, []).append(split_y)

      if page_num in suggested:
        cleaned = []
        for y in sorted(suggested[page_num]):
          if not cleaned or abs(y - cleaned[-1]) > 3:
            cleaned.append(y)
        suggested[page_num] = cleaned

    return suggested

  @staticmethod
  def _map_point_to_composite(
      x: float,
      y: float,
      transform: Dict
  ) -> Optional[Tuple[float, float]]:
    target_width = transform.get("target_width")
    target_height = transform.get("target_height")
    source_width = transform.get("source_width")
    source_height = transform.get("source_height")
    if not target_width or not target_height:
      return None
    if source_width and source_height:
      x *= target_width / source_width
      y *= target_height / source_height

    rotation_deg = transform.get("rotation_deg", 0.0)
    base_width = target_width
    base_height = target_height
    rotated_width = transform.get("rotated_width", base_width)
    rotated_height = transform.get("rotated_height", base_height)

    if rotation_deg:
      cx, cy = base_width / 2.0, base_height / 2.0
      theta = math.radians(rotation_deg)
      dx = x - cx
      dy = y - cy
      x = (dx * math.cos(theta) - dy * math.sin(theta)) + cx
      y = (dx * math.sin(theta) + dy * math.cos(theta)) + cy
      cx2, cy2 = rotated_width / 2.0, rotated_height / 2.0
      x += (cx2 - cx)
      y += (cy2 - cy)

    pad_left = transform.get("pad_left", 0)
    pad_top = transform.get("pad_top", 0)
    offset_x = transform.get("offset_x", 0)
    offset_y = transform.get("offset_y", 0)
    return (y + pad_top + offset_y, x + pad_left + offset_x)

  @staticmethod
  def _find_horizontal_line_above(
      gray: np.ndarray,
      qr_top: float,
      max_search: int,
      min_gap: int,
      dark_threshold: int = 80,
      ratio_threshold: float = 0.15,
      x_start: Optional[int] = None,
      x_end: Optional[int] = None,
      y_start: Optional[int] = None,
      y_end: Optional[int] = None
  ) -> Optional[float]:
    height, width = gray.shape
    search_end = int(max(0, qr_top - min_gap))
    search_start = int(max(0, qr_top - max_search))
    if y_start is not None:
      search_start = max(search_start, y_start)
    if y_end is not None:
      search_end = min(search_end, y_end)
    if search_end <= search_start:
      return None

    x0 = max(0, x_start) if x_start is not None else 0
    x1 = min(width, x_end) if x_end is not None else width
    if x1 <= x0:
      return None
    band = gray[search_start:search_end, x0:x1]
    if band.size == 0:
      return None

    row_dark_ratio = (band < dark_threshold).mean(axis=1)
    if row_dark_ratio.size == 0:
      return None
    max_ratio = float(row_dark_ratio.max())
    dynamic_threshold = max(ratio_threshold, max_ratio * 0.6)
    for idx in range(len(row_dark_ratio) - 1, -1, -1):
      if row_dark_ratio[idx] >= dynamic_threshold:
        return float(search_start + idx)

    return None

  @staticmethod
  def _map_rect_to_composite(
      x: float,
      y: float,
      width: float,
      height: float,
      transform: Dict
  ) -> Optional[Tuple[float, float, float, float]]:
    corners = [
      (x, y),
      (x + width, y),
      (x, y + height),
      (x + width, y + height)
    ]
    mapped = []
    for corner_x, corner_y in corners:
      mapped_point = ManualAlignmentService._map_point_to_composite(
        corner_x,
        corner_y,
        transform
      )
      if mapped_point:
        mapped.append(mapped_point)
    if not mapped:
      return None
    ys = [p[0] for p in mapped]
    xs = [p[1] for p in mapped]
    return min(ys), min(xs), max(ys), max(xs)

  def create_composite_images(
      self,
      input_files: List[Path],
      output_dir: Optional[Path] = None,
      alpha: float = 0.3,
      qr_positions_by_file: Optional[Dict[Path, Dict[int, List[Dict]]]] = None,
      progress_callback: Optional[callable] = None
  ) -> tuple[Dict[int, str], Dict[int, tuple[int, int]], Dict[Path, Dict[int, Dict]]]:
    """
        Create composite overlay images for each page number across all exams.

        Args:
            input_files: List of PDF file paths
            output_dir: Optional directory to save composite images (if None, returns base64)
            alpha: Transparency level for each page (0.3 = 30% opacity per page)

        Returns:
            Tuple of (composites, dimensions) where:
            - composites: Dict mapping page_number -> base64 image string (or file path)
            - dimensions: Dict mapping page_number -> (width, height) in pixels
        """
    if not input_files:
      return {}, {}

    log.info(f"Creating composite images from {len(input_files)} exams")

    # Determine max page count across all PDFs
    max_pages = 0
    for pdf_path in input_files:
      try:
        doc = fitz.open(str(pdf_path))
        max_pages = max(max_pages, doc.page_count)
        doc.close()
      except Exception as e:
        log.error(f"Failed to open {pdf_path.name}: {e}")
        continue

    log.info(f"Maximum pages across all exams: {max_pages}")

    # Determine target dimensions by finding the most common page size at 150 DPI
    # This ensures consistent rendering across all PDFs
    target_dimensions = self._get_target_dimensions(input_files)
    if target_dimensions:
      log.info(
        f"Target composite dimensions: {target_dimensions[0]}x{target_dimensions[1]} pixels"
      )

    # Create composite for each page number
    composites = {}
    dimensions = {}  # Track (width, height) for each composite
    transforms_by_file: Dict[Path, Dict[int, Dict]] = {}
    for pdf_path in input_files:
      transforms_by_file[pdf_path] = {}

    for page_num in range(max_pages):
      log.info(f"Creating composite for page {page_num + 1}/{max_pages}")
      if progress_callback:
        progress_callback(page_num + 1, max_pages,
                          f"Creating composites ({page_num + 1}/{max_pages})")

      anchor_question = None
      if qr_positions_by_file:
        common_questions = None
        for pdf_path in input_files:
          page_positions = qr_positions_by_file.get(pdf_path, {}).get(
            page_num, [])
          questions = {pos.get("question_number") for pos in page_positions
                       if pos.get("question_number") is not None}
          if not questions:
            continue
          if common_questions is None:
            common_questions = questions
          else:
            common_questions = common_questions & questions
        if common_questions:
          anchor_question = min(common_questions)
      log.debug(
        "Alignment page %s: anchor_question=%s",
        page_num + 1,
        anchor_question
      )

      page_entries = []

      # Collect all images for this page number
      for pdf_path in input_files:
        try:
          doc = fitz.open(str(pdf_path))

          if page_num < doc.page_count:
            page = doc[page_num]

            # Render page to image at consistent DPI
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")

            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_bytes))

            # Convert to RGB if needed (remove alpha channel)
            if img.mode != 'RGB':
              img = img.convert('RGB')

            pre_resize_width, pre_resize_height = img.size
            # Resize to target dimensions first (keeps scale consistent)
            if target_dimensions and img.size != target_dimensions:
              log.debug(
                f"Resizing {pdf_path.name} page {page_num} from {img.size} to {target_dimensions}"
              )
              img = img.resize(target_dimensions, Image.Resampling.LANCZOS)

            original_size = img.size
            rotated_size = original_size
            rotate_angle = 0.0
            page_positions = []

            if qr_positions_by_file:
              page_positions = qr_positions_by_file.get(pdf_path, {}).get(
                page_num, [])

            if anchor_question is not None and page_positions:
              angles = [pos.get("angle", 0.0) for pos in page_positions]
              angles = [angle for angle in angles if angle is not None]
              if angles:
                angles_sorted = sorted(angles)
                page_angle = angles_sorted[len(angles_sorted) // 2]
                if abs(page_angle) >= 1.0:
                  rotate_angle = page_angle
            if anchor_question is not None:
              has_anchor = False
              if page_positions:
                has_anchor = any(
                  pos.get("question_number") == anchor_question
                  for pos in page_positions
                )
              log.debug(
                "Alignment page %s pdf %s: anchor_available=%s positions=%s",
                page_num + 1,
                pdf_path.name,
                has_anchor,
                len(page_positions)
              )

            if rotate_angle:
              img = img.rotate(rotate_angle, expand=True, fillcolor='white')
              rotated_size = img.size

            anchor = None
            anchor_size = None
            if rotate_angle and anchor_question is not None and page_positions:
              try:
                from .qr_scanner import QRScanner
                qr_scanner = QRScanner()
                if qr_scanner.available:
                  scan_result = qr_scanner._scan_image_step_up(img)
                  if scan_result and scan_result["qr_codes"]:
                    qr_codes = scan_result["qr_codes"]
                    candidates = [
                      code for code in qr_codes
                      if code.get("question_number") == anchor_question
                    ]
                    if candidates:
                      anchor_code = candidates[0]
                    else:
                      anchor_code = None

                    if anchor_code:
                      rect = anchor_code["rect"]
                      scale = scan_result["scale_to_base"]
                      anchor = (
                        (rect.top + rect.height) * scale,
                        (rect.left + rect.width) * scale
                      )
                      anchor_size = max(rect.height, rect.width) * scale
              except Exception as exc:
                log.debug("QR rescan after rotation failed: %s", exc)

            if anchor is None and anchor_question is not None and page_positions:
              width_scale = 1.0
              height_scale = 1.0
              if target_dimensions and (pre_resize_width, pre_resize_height) != target_dimensions:
                width_scale = target_dimensions[0] / pre_resize_width
                height_scale = target_dimensions[1] / pre_resize_height
              anchor_pos = None
              candidates = [pos for pos in page_positions
                            if pos.get("question_number") == anchor_question]
              if candidates:
                anchor_pos = candidates[0]

              if anchor_pos:
                x_pos = anchor_pos.get("x")
                y_pos = anchor_pos.get("y")
                width_pos = anchor_pos.get("width")
                height_pos = anchor_pos.get("height")
                if x_pos is not None and y_pos is not None:
                  if width_pos is not None and height_pos is not None:
                    base_x = x_pos + width_pos
                    base_y = y_pos + height_pos
                  else:
                    base_x = x_pos
                    base_y = y_pos

                  if rotate_angle:
                    cx, cy = original_size[0] / 2.0, original_size[1] / 2.0
                    theta = math.radians(rotate_angle)
                    dx = base_x - cx
                    dy = base_y - cy
                    rotated_x = (dx * math.cos(theta) - dy * math.sin(theta)) + cx
                    rotated_y = (dx * math.sin(theta) + dy * math.cos(theta)) + cy
                    cx2, cy2 = rotated_size[0] / 2.0, rotated_size[1] / 2.0
                    base_x = rotated_x + (cx2 - cx)
                    base_y = rotated_y + (cy2 - cy)

                  x_anchor = base_x * width_scale
                  y_anchor = base_y * height_scale
                  anchor = (y_anchor, x_anchor)
                  if width_pos is not None and height_pos is not None:
                    anchor_size = max(width_pos, height_pos) * max(width_scale,
                                                                  height_scale)

            log.debug(
              "Alignment page %s pdf %s: pre_resize=%sx%s target=%s rotate=%.2f anchor=%s anchor_size=%s",
              page_num + 1,
              pdf_path.name,
              pre_resize_width,
              pre_resize_height,
              target_dimensions,
              rotate_angle,
              anchor,
              anchor_size
            )

            page_entries.append({
              "image": img,
              "anchor": anchor,
              "anchor_size": anchor_size,
              "rotation_deg": rotate_angle,
              "pdf_path": pdf_path,
              "target_width": target_dimensions[0] if target_dimensions else img.size[0],
              "target_height": target_dimensions[1] if target_dimensions else img.size[1],
              "source_width": pre_resize_width,
              "source_height": pre_resize_height,
              "rotated_width": rotated_size[0],
              "rotated_height": rotated_size[1]
            })

          doc.close()
        except Exception as e:
          log.error(
            f"Failed to process page {page_num} from {pdf_path.name}: {e}")
          continue

      if not page_entries:
        log.warning(f"No images found for page {page_num}")
        continue

      max_width = max(entry["image"].size[0] for entry in page_entries)
      max_height = max(entry["image"].size[1] for entry in page_entries)

      for entry in page_entries:
        img = entry["image"]
        width, height = img.size
        if width == max_width and height == max_height:
          entry["pad_left"] = 0
          entry["pad_top"] = 0
          entry["padded_width"] = max_width
          entry["padded_height"] = max_height
          continue
        pad_left = (max_width - width) // 2
        pad_top = (max_height - height) // 2
        padded = Image.new('RGB', (max_width, max_height), color='white')
        padded.paste(img, (pad_left, pad_top))
        entry["image"] = padded
        entry["pad_left"] = pad_left
        entry["pad_top"] = pad_top
        entry["padded_width"] = max_width
        entry["padded_height"] = max_height
        if entry["anchor"]:
          entry["anchor"] = (
            entry["anchor"][0] + pad_top,
            entry["anchor"][1] + pad_left
          )

      base_width, base_height = page_entries[0]["image"].size
      offsets = []
      ref_anchor = None
      if any(entry["anchor"] for entry in page_entries):
        anchors = [entry["anchor"] for entry in page_entries if entry["anchor"]]
        ref_anchor = max(anchors)
        shifts = []
        for entry in page_entries:
          if entry["anchor"]:
            dy = ref_anchor[0] - entry["anchor"][0]
            dx = ref_anchor[1] - entry["anchor"][1]
          else:
            dy = 0
            dx = 0
          shifts.append((dx, dy))

        anchor_sizes = [entry["anchor_size"] for entry in page_entries
                        if entry["anchor_size"]]
        cap = max(anchor_sizes) if anchor_sizes else None
        if cap is not None:
          too_large = any(
            abs(dx) > cap or abs(dy) > cap for dx, dy in shifts)
          if too_large:
            log.info(
              f"Skipping alignment on page {page_num + 1}: shift exceeds cap"
            )
            offsets = [(0, 0) for _ in page_entries]
            output_size = (base_width, base_height)
            shifts = None
        if shifts is not None:
          min_dx = min(dx for dx, _ in shifts)
          min_dy = min(dy for _, dy in shifts)
          max_dx = max(dx for dx, _ in shifts)
          max_dy = max(dy for _, dy in shifts)

          output_width = int(math.ceil(base_width + (max_dx - min_dx)))
          output_height = int(math.ceil(base_height + (max_dy - min_dy)))
          output_size = (max(output_width, base_width),
                         max(output_height, base_height))

          offsets = [
            (int(math.floor(dx - min_dx)), int(math.floor(dy - min_dy)))
            for dx, dy in shifts
          ]
      else:
        output_size = (base_width, base_height)
        offsets = [(0, 0) for _ in page_entries]

      for entry, offset in zip(page_entries, offsets):
        entry["offset_x"] = offset[0]
        entry["offset_y"] = offset[1]
        entry["canvas_width"] = output_size[0]
        entry["canvas_height"] = output_size[1]
        anchor_aligned = None
        if entry.get("anchor"):
          anchor_aligned = (
            entry["anchor"][0] + entry["offset_y"],
            entry["anchor"][1] + entry["offset_x"]
          )

        log.debug(
          "Alignment page %s pdf %s: pad_left=%s pad_top=%s offset=(%s,%s) canvas=%sx%s anchor_padded=%s anchor_aligned=%s ref_anchor=%s",
          page_num + 1,
          entry["pdf_path"].name,
          entry.get("pad_left", 0),
          entry.get("pad_top", 0),
          entry.get("offset_x", 0),
          entry.get("offset_y", 0),
          entry.get("canvas_width", base_width),
          entry.get("canvas_height", base_height),
          entry.get("anchor"),
          anchor_aligned,
          ref_anchor
        )

        pdf_path = entry["pdf_path"]
        anchor_value = entry.get("anchor")
        transforms_by_file[pdf_path][page_num] = {
          "anchor_question": anchor_question,
          "anchor_y": anchor_value[0] if anchor_value else None,
          "anchor_x": anchor_value[1] if anchor_value else None,
          "anchor_aligned_y": anchor_aligned[0] if anchor_aligned else None,
          "anchor_aligned_x": anchor_aligned[1] if anchor_aligned else None,
          "ref_anchor_y": ref_anchor[0] if ref_anchor else None,
          "ref_anchor_x": ref_anchor[1] if ref_anchor else None,
          "rotation_deg": entry.get("rotation_deg", 0.0),
          "offset_x": entry.get("offset_x", 0),
          "offset_y": entry.get("offset_y", 0),
          "pad_left": entry.get("pad_left", 0),
          "pad_top": entry.get("pad_top", 0),
          "padded_width": entry.get("padded_width", base_width),
          "padded_height": entry.get("padded_height", base_height),
          "canvas_width": entry.get("canvas_width", base_width),
          "canvas_height": entry.get("canvas_height", base_height),
          "target_width": entry.get("target_width", base_width),
          "target_height": entry.get("target_height", base_height),
          "source_width": entry.get("source_width", base_width),
          "source_height": entry.get("source_height", base_height),
          "rotated_width": entry.get("rotated_width", base_width),
          "rotated_height": entry.get("rotated_height", base_height)
        }

      # Create composite by averaging all images
      composite = self._create_overlay_composite(
        [entry["image"] for entry in page_entries],
        alpha,
        offsets=offsets,
        output_size=output_size
      )

      # Record dimensions (width, height) of this composite
      dimensions[page_num] = composite.size

      # Convert to base64 or save to file
      if output_dir:
        output_path = output_dir / f"composite_page_{page_num + 1}.png"
        composite.save(output_path)
        composites[page_num] = str(output_path)
        log.info(f"Saved composite to {output_path}")
      else:
        # Convert to base64
        buffer = io.BytesIO()
        composite.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        composites[page_num] = img_base64

    log.info(f"Created {len(composites)} composite images")
    return composites, dimensions, transforms_by_file

  def _get_target_dimensions(
      self, input_files: List[Path]) -> Optional[tuple[int, int]]:
    """
        Determine target dimensions by finding the most common page size at 150 DPI.
        This ensures all PDFs render to the same pixel dimensions regardless of source resolution.

        Args:
            input_files: List of PDF file paths

        Returns:
            Tuple of (width, height) in pixels, or None if no files
        """
    from collections import Counter

    # Collect dimensions of first page from each PDF
    dimensions_list = []
    for pdf_path in input_files:
      try:
        doc = fitz.open(str(pdf_path))
        if doc.page_count > 0:
          page = doc[0]
          pix = page.get_pixmap(dpi=150)
          dimensions_list.append((pix.width, pix.height))
        doc.close()
      except Exception as e:
        log.error(f"Failed to get dimensions from {pdf_path.name}: {e}")
        continue

    if not dimensions_list:
      return None

    # Find most common dimensions
    dimension_counts = Counter(dimensions_list)
    most_common = dimension_counts.most_common(1)[0]
    target_dims, count = most_common

    if count < len(dimensions_list):
      log.info(
        f"Found {len(dimensions_list) - count} PDFs with non-standard dimensions that will be resized"
      )

    return target_dims

  def _create_overlay_composite(self,
                                images: List[Image.Image],
                                alpha: float = 0.3,
                                offsets: Optional[List[tuple]] = None,
                                output_size: Optional[tuple] = None) -> Image.Image:
    """
        Create a composite image by overlaying multiple images with brightness-based transparency.

        Dark pixels (black ink) are rendered opaque, while lighter pixels are very transparent.
        This ensures that handwritten ink remains visible even when stacking many exams.

        Args:
            images: List of PIL Images to overlay
            alpha: Base transparency level (now modulated by pixel brightness)

        Returns:
            Composite PIL Image
        """
    if not images:
      raise ValueError("No images provided")

    # Get dimensions from first image (assume all are same size)
    width, height = images[0].size
    if offsets is None:
      offsets = [(0, 0) for _ in images]
    if output_size is None:
      output_size = (width, height)

    # Resize all images to match first image size (handle any size variations)
    resized_images = []
    for img in images:
      if img.size != (width, height):
        log.warning(f"Resizing image from {img.size} to {width}x{height}")
        img = img.resize((width, height), Image.Resampling.LANCZOS)
      resized_images.append(img)

    # Convert images to numpy arrays
    arrays = [np.array(img, dtype=np.float32) for img in resized_images]

    # Initialize composite with white background
    composite_array = np.ones(
      (output_size[1], output_size[0], 3), dtype=np.float32) * 255

    # Brightness-based alpha blending
    for arr, offset in zip(arrays, offsets):
      offset_x, offset_y = offset
      offset_x = max(0, int(offset_x))
      offset_y = max(0, int(offset_y))
      height, width = arr.shape[:2]

      x_end = min(offset_x + width, composite_array.shape[1])
      y_end = min(offset_y + height, composite_array.shape[0])
      region_width = x_end - offset_x
      region_height = y_end - offset_y

      if region_width <= 0 or region_height <= 0:
        continue

      arr_region = arr[:region_height, :region_width]
      comp_region = composite_array[offset_y:y_end, offset_x:x_end]

      # Calculate brightness for each pixel (average across RGB channels)
      brightness = np.mean(arr_region, axis=2, keepdims=True)

      # Normalize brightness to 0-1 range
      normalized_brightness = brightness / 255.0

      # Create threshold-based alpha:
      # - Very dark (< 10% brightness / > 90% dark): alpha = 1.0 (fully opaque)
      # - Moderately dark to light: smooth transition from 1.0 to 0.05
      darkness_threshold = 0.1  # 10% brightness = 90% dark
      light_threshold = 0.9  # 90% brightness = 10% dark

      # Calculate per-pixel alpha
      pixel_alpha = np.where(
        normalized_brightness < darkness_threshold,
        1.0,  # Fully opaque for very dark pixels
        np.where(
          normalized_brightness > light_threshold,
          0.05,  # Very transparent for light pixels
          # Linear interpolation for medium pixels
          1.0 - ((normalized_brightness - darkness_threshold) /
                 (light_threshold - darkness_threshold)) * 0.95))

      # Apply alpha blending with per-pixel alpha
      composite_array[offset_y:y_end, offset_x:x_end] = (
        comp_region * (1 - pixel_alpha) + arr_region * pixel_alpha)

    # Clip values to valid range and convert back to uint8
    composite_array = np.clip(composite_array, 0, 255).astype(np.uint8)

    # Convert back to PIL Image
    composite = Image.fromarray(composite_array, mode='RGB')

    return composite

  def save_split_points(self, split_points: Dict[int, List[int]],
                        output_path: Path) -> None:
    """
        Save manual split points to JSON file.

        Args:
            split_points: Dict mapping page_number -> list of y-positions
            output_path: Path to save JSON file
        """
    import json

    data = {
      "version": "1.0",
      "split_points": {
        str(k): v
        for k, v in split_points.items()
      }
    }

    with open(output_path, 'w') as f:
      json.dump(data, f, indent=2)

    log.info(f"Saved split points to {output_path}")

  def load_split_points(self, input_path: Path) -> Dict[int, List[int]]:
    """
        Load manual split points from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            Dict mapping page_number -> list of y-positions
        """
    import json

    with open(input_path, 'r') as f:
      data = json.load(f)

    # Convert string keys back to integers
    split_points = {int(k): v for k, v in data.get("split_points", {}).items()}

    log.info(
      f"Loaded split points for {len(split_points)} pages from {input_path}")
    return split_points
