# Multiple-object detection with SIFT and custom geometry

Find repeated copies of a printed milk carton in a cluttered photograph. OpenCV
provides SIFT, image loading, and drawing; NumPy implements the rest of the
pipeline. No model training or external downloads are needed.

## Run

From this directory, using Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
python -m unittest discover -s tests -v
```

The supplied configuration detects **three cartons** in `assets/scene.jpeg`.
The original photographs are preserved in `assets/`.

## Detection result

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
python run_pipeline.py --template path/to/template.jpg --scene path/to/scene.jpg --roi 100 120 250 500 --output outputs/new_scene
```

`--roi X Y W H` selects the object in the original template photograph. For an
already tightly cropped template, use `--full-template`. When overriding the
template, also override or disable the sample-specific ROI. No scene locations
or expected object count are provided to the detector.

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
