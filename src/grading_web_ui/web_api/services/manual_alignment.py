"""
Manual alignment service - creates composite images and handles user-defined split points
"""
from typing import List, Dict, Optional
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

  def create_composite_images(
      self,
      input_files: List[Path],
      output_dir: Optional[Path] = None,
      alpha: float = 0.3,
      qr_positions_by_file: Optional[Dict[Path, Dict[int, List[Dict]]]] = None,
      progress_callback: Optional[callable] = None
  ) -> tuple[Dict[int, str], Dict[int, tuple[int, int]]]:
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

            page_entries.append({
              "image": img,
              "anchor": anchor,
              "anchor_size": anchor_size
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
          continue
        pad_left = (max_width - width) // 2
        pad_top = (max_height - height) // 2
        padded = Image.new('RGB', (max_width, max_height), color='white')
        padded.paste(img, (pad_left, pad_top))
        entry["image"] = padded
        if entry["anchor"]:
          entry["anchor"] = (
            entry["anchor"][0] + pad_top,
            entry["anchor"][1] + pad_left
          )

      base_width, base_height = page_entries[0]["image"].size
      offsets = []
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
    return composites, dimensions

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
