# Multiple-object detection with SIFT and custom geometry

Find repeated copies of a printed milk carton in a cluttered photograph. OpenCV
provides SIFT, image loading, and drawing; NumPy implements the rest of the
pipeline. No model training or external downloads are needed.

## How to run the pipeline (Windows PowerShell)

### 1. Open the project folder

Open PowerShell and change to the folder containing `run_pipeline.py` and
`config.json`:

```powershell
cd "C:\Desktop\CV_Project"
```

If you saved the project elsewhere, replace the path with your project folder.
Run all commands below from this folder.

### 2. Check Python

Python **3.10 or newer** is required. Check your installed version:

```powershell
python --version
```

If `python` is not recognized, install Python and enable its **Add Python to
PATH** option, then reopen PowerShell.

### 3. Create an environment and install dependencies (first run only)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This installs NumPy and OpenCV in the project's `.venv` folder. The commands
below use that environment directly, so no activation command is needed.
Internet access is needed to download dependencies on the first installation.

### 4. Check the input images and run

The supplied images should be at:

```text
assets/
  template.jpeg
  scene.jpeg
```

The default `config.json` already contains their paths and the template crop.
Run:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --config config.json
```

Wait for the terminal to print the detection count and output folder. With the
supplied photographs, the reference run detects **three cartons**. It also
prints the feature counts, supporting inliers, and alignment error for each
detection. The original photographs are preserved.

### 5. View the results

The pipeline creates `outputs/` automatically. Open the final detection image:

```powershell
Invoke-Item .\outputs\detections.jpg
```

The main output files are:

- `outputs/detections.jpg`: the scene with a boundary around each detected carton.
- `outputs/naive_matches.jpg`: tentative feature matches before geometric verification.
- `outputs/template_crop.jpg`: the cropped template used by the pipeline.
- `outputs/results.json`: detection statistics, transformations, and run settings.

To run again, repeat step 4. Files in the chosen output folder are overwritten.

### Optional: run the automated tests

Tests are separate from running the image pipeline:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

A successful run ends with `Ran 7 tests` and `OK`.

## Input images and detection result

### 1. Template image

The isolated milk carton used as the reference object.

Image path: `assets/template.jpeg`

![Original template photograph of the milk carton](assets/template.jpeg)

### 2. Scene image

The cluttered photograph containing three partly occluded milk cartons.

Image path: `assets/scene.jpeg`

![Original scene photograph containing three milk cartons](assets/scene.jpeg)

### 3. Final detections

The image below shows the three detected milk cartons with their estimated boundaries.

Image path: `outputs/detections.jpg`

![Three detected milk cartons in the cluttered scene](outputs/detections.jpg)

[Open the full-size detection image](outputs/detections.jpg)

## Files

| File | Purpose |
| --- | --- |
| `run_pipeline.py` | CLI, SIFT extraction, output images, timing, and metrics |
| `vision.py` | Exact descriptor matching, ratio test, 4D Hough voting, affine RANSAC, greedy extraction, polygon IoU NMS |
| `config.json` | Input paths, template crop, thresholds, random seed |
| `tests/test_vision.py` | Synthetic ground-truth and edge-case checks |
| `REPORT.md` | Short assignment report with matching/final comparison |
| `outputs/detections.jpg` | Scene with final boundaries and object labels |
| `outputs/naive_matches.jpg` | Up to 160 tentative correspondences, before geometry |
| `outputs/template_crop.jpg` | Template used for feature extraction |
| `outputs/results.json` | Affines, corners, support, errors, configuration, timing, and extraction diagnostics |

## Using different photographs

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --template "path/to/template.jpg" --scene "path/to/scene.jpg" --roi 100 120 250 500 --output outputs/new_scene
```

`--roi X Y W H` selects the object in the original template photograph. For an
already tightly cropped template, use `--full-template`. When overriding the
template, also override or disable the sample-specific ROI. No scene locations
or expected object count are provided to the detector.

For example, with a tightly cropped template:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --template "path/to/cropped_template.jpg" --scene "path/to/scene.jpg" --full-template --output outputs/new_scene
```

Replace the example image paths and crop coordinates with your own values.

Paths inside a configuration are relative to that configuration's directory.
CLI path overrides are relative to the working directory. Both photographs run
at their original resolution; the supplied ROI is `[592, 306, 241, 560]`.
Affines map **cropped-template coordinates** into original scene coordinates.
To map original-template points, subtract the ROI origin first.

## Algorithm

1. Extract positions, scales, orientations, and 128-dimensional SIFT descriptors.
2. Compute every scene-to-template Euclidean distance in memory-bounded blocks.
   Accept a nearest neighbor only if its first/second distance ratio is below
   0.78. Many scene features can match the same template feature.
3. Predict center, relative scale, and rotation from each match. Vote into a
   sparse 4D accumulator with logarithmic scale and circular angle bins. Adjacent
   bins receive votes to reduce sensitivity to bin boundaries.
4. Try candidate clusters by decreasing vote count. Sample three matches,
   reject degenerate triangles, fit an affine with centered least squares, and
   score independent inliers. RANSAC has a 1,200-trial cap and adaptively reduces
   trials using a nominal 99.5% success target (not an accuracy guarantee).
5. Refit and recruit consistent matches from all remaining correspondences.
   Reject implausible scale, reflection, distortion, inadequate spatial support,
   and boundaries mostly outside the image. Accept at least eight independent
   points. Remove only geometrically consistent matches and repeat until all
   remaining candidate clusters fail; there is no fixed detection count.
6. Rank detections by inlier count then median error. Use independently
   implemented convex polygon clipping and IoU to suppress duplicates. Transform
   the crop's four corners to draw the final boundaries.

Tests cover exact matching against brute force, reuse of template descriptors,
ambiguous/empty descriptors, affine recovery and degeneracy, RANSAC with 60%
outliers, polygon overlap and NMS, angular wraparound, three-instance extraction
with different rotations/scales, and empty detections. See the report for actual
photograph results and limitations.
