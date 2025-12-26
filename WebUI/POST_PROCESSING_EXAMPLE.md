# Post-Processing Submissions - Usage Guide

## Overview

The `ExamProcessor.post_process_submissions()` method allows you to apply operations across all submissions for each problem number. This is perfect for statistical analysis like population-based blank detection.

## Key Concepts

1. **DTOs have image access methods** - No need to manually decode base64
2. **Operations modify problems in-place** - Changes persist in the parent submission
3. **Grouped by problem number** - All problem 1s together, all problem 2s together, etc.

## ProblemDTO Image Methods

```python
from web_api.dtos import ProblemDTO

problem = ProblemDTO(...)

# Get as PIL Image (RGB)
img = problem.get_image()
print(img.size)  # (width, height)

# Get as grayscale PIL Image
gray_img = problem.get_grayscale_image()

# Get as numpy array
pixels = problem.get_image_array(grayscale=True)  # 2D array
pixels_rgb = problem.get_image_array(grayscale=False)  # 3D array (H, W, 3)

# Calculate black pixel ratio (built-in!)
ratio = problem.calculate_black_pixel_ratio(threshold=200)
print(f"Black pixel ratio: {ratio:.4f}")
```

## Basic Usage

### Example 1: Population-Based Blank Detection

```python
from web_api.services.exam_processor import ExamProcessor
from web_api.services.blank_detection import population_blank_detection

# Process exams
processor = ExamProcessor()
matched, unmatched = processor.process_exams(
    input_files=pdf_files,
    canvas_students=students,
    manual_split_points=split_points
)

# Apply population blank detection to all problems
processor.post_process_submissions(
    submissions=unmatched,
    operations=[
        lambda num, probs: population_blank_detection(num, probs, percentile_threshold=5.0)
    ]
)

# Now problems are marked blank - ready to save
for submission in unmatched:
    for problem in submission.problems:
        if problem.is_blank:
            print(f"Problem {problem.problem_number}: {problem.blank_reasoning}")
```

### Example 2: Custom Operation

```python
import numpy as np

def detect_outliers(problem_number: int, problems: List[ProblemDTO]):
    """Mark problems with unusually low ink as blank"""

    # Get black pixel ratios using built-in method
    ratios = [p.calculate_black_pixel_ratio() for p in problems]

    # Calculate statistics
    mean = np.mean(ratios)
    std = np.std(ratios)
    threshold = mean - 2 * std  # 2 standard deviations below mean

    # Mark outliers
    for i, problem in enumerate(problems):
        if ratios[i] < threshold:
            problem.mark_blank(
                confidence=0.90,
                method="outlier",
                reasoning=f"Ratio {ratios[i]:.4f} < threshold {threshold:.4f} (mean={mean:.4f}, std={std:.4f})"
            )

# Use it
processor.post_process_submissions(
    unmatched,
    operations=[detect_outliers]
)
```

### Example 3: Multiple Operations

```python
def detect_blanks(problem_number: int, problems: List[ProblemDTO]):
    """Population-based blank detection"""
    ratios = [p.calculate_black_pixel_ratio() for p in problems]
    threshold = np.percentile(ratios, 5)

    for i, problem in enumerate(problems):
        if ratios[i] < threshold:
            problem.mark_blank(0.95, "population", f"Ratio: {ratios[i]:.4f}")

def normalize_scores(problem_number: int, problems: List[ProblemDTO]):
    """Normalize scores based on max_points"""
    # Get max points from metadata or first problem
    max_pts = problems[0].max_points
    if max_pts:
        for problem in problems:
            problem.max_points = max_pts  # Ensure all have same max

def log_statistics(problem_number: int, problems: List[ProblemDTO]):
    """Log statistics about this problem"""
    ratios = [p.calculate_black_pixel_ratio() for p in problems]
    blank_count = sum(1 for p in problems if p.is_blank)

    print(f"Problem {problem_number}:")
    print(f"  Submissions: {len(problems)}")
    print(f"  Blank: {blank_count} ({blank_count/len(problems)*100:.1f}%)")
    print(f"  Ink ratio: mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}")

# Apply all operations in order
processor.post_process_submissions(
    unmatched,
    operations=[
        detect_blanks,
        normalize_scores,
        log_statistics
    ]
)
```

## Advanced: Working with Images

### Example 4: Image Processing

```python
import cv2
import numpy as np
from PIL import Image

def detect_handwriting(problem_number: int, problems: List[ProblemDTO]):
    """Detect handwriting using computer vision"""

    for problem in problems:
        # Get image as numpy array
        img = problem.get_image_array(grayscale=True)

        # Apply thresholding
        _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count significant contours (handwriting strokes)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 10]

        if len(significant_contours) < 5:
            problem.mark_blank(
                confidence=0.85,
                method="contour",
                reasoning=f"Only {len(significant_contours)} contours detected"
            )

processor.post_process_submissions(unmatched, operations=[detect_handwriting])
```

### Example 5: Caching Calculations

```python
def detect_with_caching(problem_number: int, problems: List[ProblemDTO]):
    """Cache expensive calculations"""

    # Calculate once, store in list
    ratios = []
    for problem in problems:
        # This calculation is cached internally if done multiple times
        ratio = problem.calculate_black_pixel_ratio()
        ratios.append(ratio)

    # Now use cached ratios
    threshold = np.percentile(ratios, 5)

    for i, problem in enumerate(problems):
        if ratios[i] < threshold:
            problem.mark_blank(0.95, "population", f"Ratio: {ratios[i]:.4f}")
```

## Integration in routes/uploads.py

```python
from web_api.services.exam_processor import ExamProcessor
from web_api.services.blank_detection import population_blank_detection

# Process exams
processor = ExamProcessor(ai_provider=ai_provider)
matched, unmatched = processor.process_exams(...)

# Apply blank detection BEFORE saving to database
processor.post_process_submissions(
    submissions=unmatched,  # Or matched + unmatched
    operations=[
        lambda num, probs: population_blank_detection(num, probs, percentile_threshold=5.0)
    ]
)

# Now save to database with blank flags already set
with with_transaction() as repos:
    for sub_dto in unmatched:
        # Create submission
        submission = Submission(...)
        repos.submissions.create(submission)

        # Create problems with blank flags
        for prob_dto in sub_dto.problems:
            problem = Problem(
                ...
                is_blank=prob_dto.is_blank,  # Already set by post-processing!
                blank_confidence=prob_dto.blank_confidence,
                blank_method=prob_dto.blank_method,
                blank_reasoning=prob_dto.blank_reasoning
            )
            repos.problems.create(problem)
```

## Benefits

1. **Type Safety**: `problem.get_image()` is type-checked, no manual base64 decoding
2. **Convenient**: Built-in methods like `calculate_black_pixel_ratio()`
3. **In-Place**: Modifications persist automatically
4. **Flexible**: Pass any function that takes `(int, List[ProblemDTO])`
5. **Composable**: Chain multiple operations
6. **Error Handling**: Each operation is wrapped in try/except

## Available Image Methods on ProblemDTO

| Method | Returns | Description |
|--------|---------|-------------|
| `get_image()` | `PIL.Image` (RGB) | Full color image |
| `get_grayscale_image()` | `PIL.Image` (L) | Grayscale image |
| `get_image_array(grayscale=False)` | `np.ndarray` | Numpy array (2D or 3D) |
| `calculate_black_pixel_ratio(threshold=200)` | `float` | Ratio of dark pixels (0.0-1.0) |

## Tips

1. **Use lambdas for partial application**:
   ```python
   operations=[
       lambda n, p: population_blank_detection(n, p, percentile_threshold=3.0)
   ]
   ```

2. **Operations modify in-place - no return needed**:
   ```python
   def my_operation(num, problems):
       for p in problems:
           p.is_blank = True  # ✅ Persists
   ```

3. **Access all DTO helper methods**:
   ```python
   def my_operation(num, problems):
       for p in problems:
           p.mark_blank(0.95, "custom", "reason")  # ✅ Uses helper
   ```

4. **Group by problem number is automatic**:
   ```python
   # post_process_submissions handles grouping
   # Your function receives all problem 1s, then all problem 2s, etc.
   ```
